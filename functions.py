# imports
import os
import re
import bz2
import cv2
import git
import gzip
import torch
import locale
import pickle
import random
import shutil
import struct
import numpy as np
import ultralytics
import pandas as pd
from os import listdir
from ultralytics import YOLO
from matplotlib import pyplot
from json import loads, dumps
import xml.etree.ElementTree as ET
import torchvision.models as models

# change image path to current repo
def repo_image_path(path_from_repo_root):
    repo = git.Repo('.', search_parent_directories=True)
    repo_root = repo.working_tree_dir
    relative_path = repo_root + path_from_repo_root
    return relative_path

# function to predict and plot image
"""split to predict and print"""
def predict_plot_image(image_path,model_trained):
  results = model_trained(image_path)
  res_plotted = results[0].plot()
  cv2.imshow('image',res_plotted)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

""" Functions for prediction """

# returns masks, boxes and class probabilities for each image
def return_bbox_masks_probs(res_lst):
    for result in res_lst:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmenation masks outputs
        probs = result.probs  # Class probabilities for classification outputs
    return boxes, masks, probs

# create dataset with image data, predicted boxes and size
def create_df(dataset_dir, image_type, yolo_model, color=None):
    pred_df = pd.DataFrame()

    # iterate over images
    for image in (os.listdir(dataset_dir)):
        if (image.endswith(image_type)):
            # load and prepare image
            photo_filename = dataset_dir + "/" + image
            # load image in GBR format
            im = cv2.imread(photo_filename)
            h, w, _ = im.shape

            if color == "BW":
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            # make prediction for the image
            results = yolo_model(im)

            # save bbox, masks, probabilities
            boxes, masks, _ = return_bbox_masks_probs(results)
            list_of_boxes = boxes.xyxy.tolist()

            # append to dataframe
            df = pd.DataFrame([{'name': image, 'image': photo_filename, 'height': h, 'width': w, 'boxes': list_of_boxes}])
            pred_df = pd.concat([pred_df, df], ignore_index=True)
    return pred_df

# extract predicted boxes from annotation file
def extract_boxes(dataset_dir):
    boxes = {}

    for txt_file in (os.listdir(dataset_dir)):
        # load and prepare image
        photo_filename = dataset_dir + "/" + txt_file

        with open(photo_filename, 'r') as file:
            bbox_list = []
            for line in file.readlines():
                # Split the line into a list of words
                words = line.strip().split()

                # Extract the label and bounding box coordinates
                label = words[0]
                bbox = [float(x) for x in words[1:]]

                # Add the bounding box to the list of bounding boxes
                bbox_list.append(bbox)
            file.close()
        boxes[txt_file] = bbox_list
    return boxes

# extract real boxes from annotation file in xml format
def extract_xml_boxes(dataset_dir):
    boxes = {}

    for txt_file in (os.listdir(dataset_dir)):
        # load and prepare image
        photo_filename = dataset_dir + "/" + txt_file

        with open(photo_filename, 'r') as file:
            tree = ET.parse(photo_filename)
            root = tree.getroot()
            bbox_list = []
            for neighbor in root.iter('bndbox'):
                xmin = (neighbor.find('xmin').text)
                ymin = (neighbor.find('ymin').text)
                xmax = (neighbor.find('xmax').text)
                ymax = (neighbor.find('ymax').text)
                bbox_list.append([xmin, ymin, xmax, ymax])
        boxes[txt_file] = bbox_list
    return boxes

# create dataframe with image name and annotations
def create_annotations_df(annotations_dir):
    anno_dict = extract_boxes(annotations_dir)
    df_anno = pd.DataFrame({'name': anno_dict.keys(), 'annotations': anno_dict.values()})
    return df_anno

# convert absolute boxes measures to relative format
def boxes_abs_to_relative(boxes, h, w):
    relative_boxes = []
    if boxes:
        for box in boxes:
            xmin = float(box[0]) / w
            ymin = float(box[1]) / h
            xmax = float(box[2]) / w
            ymax = float(box[3]) / h
            relative_boxes.append([xmin, ymin, xmax, ymax])
    return relative_boxes

# convert yolo boxes measures to relative format
def yolo_to_relative(boxes):
    relative_boxes = []
    for box in boxes:
        xmin = (box[0] - box[2] / 2)
        ymin = (box[1] - box[3] / 2)
        xmax = (box[0] + box[2] / 2)
        ymax = (box[1] + box[3] / 2)
        relative_boxes.append([xmin, ymin, xmax, ymax])
    return relative_boxes

# calculate and return IOU per image
def bbox_iou(box1, box2):
    # box1 and box2 are lists with 4 elements [xmin, ymin, xmax, ymax]
    # Calculate the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate the area of the intersection rectangle
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate the union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate the IoU
    iou = intersection_area / union_area
    return iou

# function to calculate IoU between two lists of bboxes
def calculate_iou_list(pred_bboxes, ann_bboxes):
    iou_list = []

    for ann_bbox in ann_bboxes:
        if type(ann_bbox) == "<class 'float'>":
            continue
        iou_row = []
        for pred_bbox in pred_bboxes:
            iou = bbox_iou(pred_bbox, ann_bbox)
            iou_row.append(iou)
        iou_list.append(iou_row)

    return iou_list

# function to calculate the maximum IOU in case of
# #several predictions for the same image
def max_iou(list_of_iou):
    max_lst = []
    for lst in list_of_iou:
        if lst:
            max_lst.append(max(lst))
        else:
            max_lst.append(0)
    return max_lst

""" Pipeline function """

# create dataframe with all the relevant data
def pipeline(dataset_name, dataset_path, annotation_path, image_format, model, annotation_foramt=None, color=None):
    df_images = create_df(dataset_path, image_format, model, color)

    df_images['relative_boxes'] = df_images.apply(
        lambda row: boxes_abs_to_relative(row['boxes'], row['height'], row['width']), axis=1)

    if annotation_foramt == '.xml':
        annotaions_dict = extract_xml_boxes(annotation_path)
    else:
        annotaions_dict = extract_boxes(annotation_path)
    df_annotations = pd.DataFrame({'names': annotaions_dict.keys(), 'annotations': annotaions_dict.values()})

    df_annotations['names'] = df_annotations.apply(lambda row: re.sub('txt$', 'jpg', row['names']), axis=1)
    df_annotations['names'] = df_annotations.apply(lambda row: re.sub('xml$', 'jpg', row['names']), axis=1)

    df_images = df_images.set_index('name').join(df_annotations.set_index('names'))

    # remove empty annotations
    df_images['anno_type'] = df_images.apply(lambda row: type(row['annotations']), axis=1)
    df_images = df_images.loc[df_images['anno_type'] != float]

    if dataset_name == 'coco128':
        df_images['relative_annotations'] = df_images.apply(lambda row: yolo_to_relative(row['annotations']), axis=1)
    else:
        df_images['relative_annotations'] = df_images.apply(lambda row: boxes_abs_to_relative(row['annotations'], row['height'], row['width']), axis=1)

    df_images['iou_score'] = df_images.apply(
        lambda row: calculate_iou_list(row['relative_boxes'], row['relative_annotations']), axis=1)

    df_images['max_iou_score'] = df_images.apply(lambda row: max_iou(row['iou_score']), axis=1)

    df_images['num_of_annotations'] = df_images.apply(lambda row: len(row['annotations']), axis=1)

    df_images['avg_score'] = df_images.apply(lambda row: sum(row['max_iou_score']) / row['num_of_annotations'], axis=1)

    total_iou = df_images['avg_score'].mean()
    return df_images, total_iou

# functions to print images with low iou score
def print_low_score_images(df_low, dataset_name,model_trained):
  image_list = df_low.index.values.tolist()
  for image in image_list:
    print_image_by_dataset_and_name(image, dataset_name,model_trained)

#functions to save pkl files
def print_image_by_dataset_and_name(image, data_set_name,model):
    path = f"/{data_set_name}/{image}"
    repo_path = repo_image_path(path)
    predict_plot_image(repo_path,model)


def save_zipped_pickle(obj, filename, protocol=-1):
    with bz2.BZ2File(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)

def load_zipped_pickle_to_dataframe(filename):
    with bz2.BZ2File(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        df = pd.DataFrame(loaded_object)  # Assuming the loaded object is compatible with DataFrame creation
        return df
