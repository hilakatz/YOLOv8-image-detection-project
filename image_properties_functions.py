''' image properties functions '''

# imports
import math
import os
import cv2
import skimage
from cv2 import IMREAD_COLOR, IMREAD_UNCHANGED
from PIL import Image, ImageStat
from scipy.signal import convolve2d
from PIL import Image, ImageFilter

import functions as utils

import numpy as np

import matplotlib.image as img
import matplotlib.pyplot as plt
from scipy.cluster.vq import whiten
from scipy.cluster.vq import kmeans
import pandas as pd
import functions as utils


# convert BGR to RGB
def BGR2RGB(BGR_img):
  # turning BGR pixel color to RGB
  rgb_image = cv2.cvtColor(BGR_img, cv2.COLOR_BGR2RGB)
  return rgb_image


# Convert the image to grayscale
def convert_image_to_grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


""" aspect ratio (width-height) """
def return_aspect_ratio(w,h):
    return float(w) / h


"""brightness"""
#source: https://stackoverflow.com/questions/3490727/what-are-some-methods-to-analyze-image-brightness-using-python

# Calculate the mean brightness value
def get_image_brightness(image_path):
    path = utils.repo_image_path(image_path)
    image = cv2.imread(path)

    im = convert_image_to_grayscale(image)
    brightness = int(round(cv2.mean(im)[0]))
    return brightness


""" contrast """
def get_image_contrast(image_path):
    path = utils.repo_image_path(image_path)
    image = cv2.imread(path)
    # load image as YUV (or YCbCR) and select Y (intensity)
    y = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)[:, :, 0]

    # compute min and max of Y
    min_y = np.min(y)
    max_y = np.max(y)

    # convert to float
    min_y = min_y.astype(np.float64)
    max_y = max_y.astype(np.float64)

    # compute contrast
    contrast = (max_y - min_y) / (max_y + min_y)
    return contrast


"""blur
https://www.kaggle.com/code/eladhaziza/perform-blur-detection-with-opencv
"""

#define bluriness using laplacian
def get_image_sharpness(image_path):
    path = utils.repo_image_path(image_path)
    #read the image
    image = cv2.imread(path)
    #Compute Laplacian
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    #calculate the variance of the laplaican
    var = np.var(laplacian)
    return var


""" object precentage from image """
#considering there is at least one object per image

def object_percentage(image_path, object_color_lower, object_color_upper):

    # Read the image
    img = cv2.imread(image_path)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create a mask for the object color range
    mask = cv2.inRange(hsv, object_color_lower, object_color_upper)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the total image area
    total_area = img.shape[0] * img.shape[1]

    object_percentages = []
    #assuming at least one object
    for contour in contours:
        # Calculate the object area
        object_area = cv2.contourArea(contour)

        # Calculate the object percentage
        object_percentage = (object_area / total_area) * 100
        object_percentages.append(object_percentage)

    return object_percentages


def edge_detection(image_path, lower_threshold=50, higher_threshold=150):
    path = utils.repo_image_path(image_path)
    image = cv2.imread(path)
    height, width = image.shape[:2]

    gray = convert_image_to_grayscale(image)

    # Apply Canny edge detection algorithm
    edges = cv2.Canny(gray, lower_threshold, higher_threshold)

    # Count the number of edges
    num_edges = cv2.countNonZero(edges)

    # Return the edge frequency
    return num_edges / (height * width)


def get_image_noise(image_path):
    path = utils.repo_image_path(image_path)
    # Load the image
    img = cv2.imread(path)
    # Convert the image to grayscale
    gray = convert_image_to_grayscale(img)
    # Calculate the standard deviation of pixel intensities as a measure of image noise
    noise = np.std(gray)
    return noise


def get_image_saturation(image_path):
    path = utils.repo_image_path(image_path)
    # Load the image
    img = cv2.imread(path)
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Calculate the standard deviation of pixel intensities as a measure of image noise
    saturation = np.std(hsv[:, :, 1])
    return saturation


def get_image_entropy(image_path):
    path = utils.repo_image_path(image_path)
    # Load the image
    img = cv2.imread(path)
    # Convert the image to grayscale
    gray = convert_image_to_grayscale(img)
    # Calculate the entropy of the grayscale image
    entropy = skimage.measure.shannon_entropy(gray)
    return entropy


def estimate_noise(image_path):
    # https://stackoverflow.com/questions/2440504/noise-estimation-noise-measurement-in-image
    path = utils.repo_image_path(image_path)
    # Load the image
    img = cv2.imread(path)
    # Convert the image to grayscale
    gray = convert_image_to_grayscale(img)
    H, W = gray.shape

    M = [[1, -2, 1],
        [-2, 4, -2],
        [1, -2, 1]]

    sigma = np.sum(np.sum(np.absolute(convolve2d(gray, M))))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

    return sigma


#   return color_channel_percentage
def get_channel_percentage(image_path, channel):
    path = utils.repo_image_path(image_path)
    image = cv2.imread(path)

    # Get the sum of all pixels in the image (across all channels).
    total_brightness = np.sum(image)

    # Get the sum of the pixels in the specified channel.
    channel_sum = np.sum(image[:, :, channel])

    # Calculate the percentage of the specified channel relative to the total brightness.
    channel_percentage = channel_sum / total_brightness * 100

    return channel_percentage


def get_salt_and_pepper_noise(image_path):
    path = utils.repo_image_path(image_path)
    # Load the image
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Calculate the percentage of white and black pixels
    num_pixels = img.shape[0] * img.shape[1]
    white_pixels = np.sum(img == 255)
    black_pixels = np.sum(img == 0)
    salt_pepper_pixels = white_pixels + black_pixels
    noise_percentage = salt_pepper_pixels / num_pixels

    # Apply threshold to identify salt and pepper noise
    return noise_percentage

def get_image_blurriness_by_model(image_path, model):
    # Load and preprocess the image
    path = utils.repo_image_path(image_path)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (600, 600))  # Assuming your CNN model requires input size of (600, 600)
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make the prediction
    prediction = model.predict(image)
    predicted_class = int(np.round(prediction[0][0]))  # Convert prediction to binary value (0 or 1)

    return predicted_class

#find dominant color in an image 
#https://www.geeksforgeeks.org/extract-dominant-colors-of-an-image-using-python/
#TODO: find the most common combi of colors 
#TODO: add to the dashboard
#TODO: make a boxplot of the different combis with a certain range

def dominant_colors(image_path):
    path = utils.repo_image_path(image_path)
    image = cv2.imread(path)
    # print(image.shape)

    # Store RGB values of all pixels in lists r, g and b
    r = []
    g = []
    b = []

    for row in image:
        for temp_r, temp_g, temp_b in row:
            r.append(temp_r)
            g.append(temp_g)
            b.append(temp_b)
    
    df = pd.DataFrame({'red' : r,
                            'green' : g,
                            'blue' : b})
    
    #scale the DataFrame to get standardized values
    df['scaled_color_red'] = whiten(df['red'])
    df['scaled_color_blue'] = whiten(df['blue'])
    df['scaled_color_green'] = whiten(df['green'])

    #find the number of clusters in k-means using the elbow plot approach
    # create a list of distortions from the kmeans function
    cluster_centers, _ = kmeans(df[['scaled_color_red',
                                        'scaled_color_blue',
                                        'scaled_color_green']], 3)
    
    dominant_colors = []
    
    red_std, green_std, blue_std = df[['red',
                                            'green',
                                            'blue']].std()
    
    #Standardized value = Actual value / Standard Deviation
    # Get standard deviations of each color
    for cluster_center in cluster_centers:
        red_scaled, green_scaled, blue_scaled = cluster_center
        dominant_colors.append((
            red_scaled * red_std / 255,
            green_scaled * green_std / 255,
            blue_scaled * blue_std / 255
        ))
    
    return (dominant_colors)
    # plt.imshow([dominant_colors])
    # plt.show()


#check for occlusion
# TODO: check how to get the annotaions into boxes  
def check_occlusion(boxes):
    max_iou = 0 
    for i in range(len(boxes)):
        for j in range(i +1, len(boxes)):
            iou_percentage = utils.bbox_iou(boxes[i], boxes[j])
            max_iou = max(max_iou, iou_percentage)
    
    return max_iou

def blur_some_photos(image_path, name):
    path = utils.repo_image_path(image_path)
    image = cv2.imread(path)
    image = Image.open(path)
    # Blur the photo.
    blurred_image = image.filter(ImageFilter.BLUR)

    # Open folder to save the blurred image.
    if not os.path.exists("Zebra_modified"):
        os.makedirs("Zebra_modified")

    # Save the resulting image.

    blurred_image.save(f"Zebra_modified/{name}")
    return f"/Zebra_modified/{name}"