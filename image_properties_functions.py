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

def dominant_colors(image_path):
    import matplotlib.image as img
    import matplotlib.pyplot as plt
    from scipy.cluster.vq import whiten
    from scipy.cluster.vq import kmeans
    import pandas as pd
    
    path = utils.repo_image_path(image_path)
    image = cv2.imread(path)

    r = []
    g = []
    b = []
    for row in image:
        for temp_r, temp_g, temp_b, temp in row:
            r.append(temp_r)
            g.append(temp_g)
            b.append(temp_b)
    
    df = pd.DataFrame({'red' : r,
                            'green' : g,
                            'blue' : b})
    
    df['scaled_color_red'] = whiten(df['red'])
    df['scaled_color_blue'] = whiten(df['blue'])
    df['scaled_color_green'] = whiten(df['green'])
    
    cluster_centers, _ = kmeans(df[['scaled_color_red',
                                        'scaled_color_blue',
                                        'scaled_color_green']], 3)
    
    dominant_colors = []
    
    red_std, green_std, blue_std = df[['red',
                                            'green',
                                            'blue']].std()
    
    for cluster_center in cluster_centers:
        red_scaled, green_scaled, blue_scaled = cluster_center
        dominant_colors.append((
            red_scaled * red_std / 255,
            green_scaled * green_std / 255,
            blue_scaled * blue_std / 255
        ))
    
    print(dominant_colors)
    plt.imshow([dominant_colors])
    plt.show()




import numpy as np
import itertools

def check_occlusion(bounding_boxes):
    """Checks if there is occlusion between objects in an image.

    Args:
        bounding_boxes: A list of bounding boxes.

    Returns:
        The percentage of occlusion between objects in an image.
    """

    array_of_combinations = list(itertools.combinations(bounding_boxes, 2))
    percentage_of_occlusion = 0
    for combination in array_of_combinations:
        intersection_area = calculate_intersection_area(combination[0], combination[1])
        union_area = calculate_union_area(combination[0], combination[1])
        percentage_of_occlusion += intersection_area / union_area * 100

    if percentage_of_occlusion > 0:
        return percentage_of_occlusion / len(array_of_combinations)
    else:
        return 0

#the following 3 functions are for occlusion 
def calculate_intersection_area(bounding_box1, bounding_box2):
    """Calculates the intersection area between two bounding boxes.

    Args:
        bounding_box1: A bounding box.
        bounding_box2: A bounding box.

    Returns:
        The intersection area between two bounding boxes.
    """

    intersection_top = max(bounding_box1[0], bounding_box2[0])
    intersection_left = max(bounding_box1[1], bounding_box2[1])
    intersection_bottom = min(bounding_box1[2], bounding_box2[2])
    intersection_right = min(bounding_box1[3], bounding_box2[3])
    if intersection_bottom <= intersection_top or intersection_right <= intersection_left:
        return 0
    else:
        return (intersection_bottom - intersection_top) * (intersection_right - intersection_left)

def calculate_union_area(bounding_box1, bounding_box2):
    """Calculates the union area between two bounding boxes.

    Args:
        bounding_box1: A bounding box.
        bounding_box2: A bounding box.

    Returns:
        The union area between two bounding boxes.
    """

    union_area = calculate_area(bounding_box1) + calculate_area(bounding_box2) - calculate_intersection_area(bounding_box1, bounding_box2)
    return union_area

def calculate_area(bounding_box):
    """Calculates the area of a bounding box.

    Args:
        bounding_box: A bounding box.

    Returns:
        The area of a bounding box.
    """

    return (bounding_box[2] - bounding_box[0]) * (bounding_box[3] - bounding_box[1])
