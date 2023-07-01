''' image properties functions '''

# imports
import math

import cv2
import skimage
from cv2 import IMREAD_COLOR, IMREAD_UNCHANGED
from PIL import Image, ImageStat
from scipy.signal import convolve2d

import functions as utils

import numpy as np


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

    gray = convert_image_to_grayscale(image)


    # Apply Canny edge detection algorithm
    edges = cv2.Canny(gray, lower_threshold, higher_threshold)

    # Count the number of edges
    num_edges = cv2.countNonZero(edges)

    # Print the number of edges
    return num_edges


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
    height, width = image.shape[:2]

    # Get the sum of the pixels in the red channel.
    channel_sum = 0
    for row in range(height):
        for col in range(width):
            channel_sum += image[row, col, channel]

    # Calculate the percentage of the red channel.
    channel_percentage = channel_sum / (width * height)

    return channel_percentage
