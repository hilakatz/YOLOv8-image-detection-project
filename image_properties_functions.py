''' image properties functions '''

# imports
import cv2
from cv2 import IMREAD_COLOR, IMREAD_UNCHANGED
from PIL import Image, ImageStat

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
def get_image_brightness(image):
    im = convert_image_to_grayscale(image)
    brightness = int(round(cv2.mean(im)[0]))
    return brightness

# calculate with rms?

""" perceived brightness """



""" contrast """
def get_image_contrast(image):
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


#def get_image_contrast(image):
#    # Calculate the standard deviation of pixel intensities
#    contrast = np.std(image)
#    return contrast


""" BGR histograms """
def bgr_histograms(image, name):

    # Get BGR data from image
    blue_channel = cv2.calcHist([img], [0], None, [256], [0, 256])
    green_channel = cv2.calcHist([img], [1], None, [256], [0, 256])
    red_channel = cv2.calcHist([img], [2], None, [256], [0, 256])

    # Separate Histograms for each color
    plt.subplot(3, 1, 1)
    plt.title("Histogram of Blue Image")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Pixel Frequency")
    plt.plot(blue_channel, color="blue")

    plt.subplot(3, 1, 2)
    plt.title("Histogram of Green Image")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Pixel Frequency")
    plt.plot(green_channel, color="green")

    plt.subplot(3, 1, 3)
    plt.title("Histogram of Red Image")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Pixel Frequency")
    plt.plot(red_channel, color="red")

    hist_name = 'rgb_histogram ' + name

    plt.savefig(hist_name)




"""blur
https://www.kaggle.com/code/eladhaziza/perform-blur-detection-with-opencv
"""

#define bluriness using laplacian
def is_blurry(image):
  #read the image
  image = image

  #Compute Laplacian 
  laplacian = cv2.Laplacian(image, cv2.CV_64F)

  #calculate the variance of the laplaican 
  var = np.var(laplacian)

  return var

def variance_of_laplacian(img2):
  # compute the Laplacian of the image and then return the focus
  # measure, which is simply the variance of the Laplacian
  gray = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
  return cv2.Laplacian(gray, cv2.CV_64F).var()




def blurrinesDetection(directories, threshold):
  columns = 3
  rows = len(directories) // 2
  fig = plt.figure(figsize=(5 * columns, 4 * rows))
  for i, directory in enumerate(directories):
    fig.add_subplot(rows, columns, i + 1)
    img = cv2.imread(directory)
    text = "Not Blurry"
    # if the focus measure is less than the supplied threshold,
    # then the image should be considered "blurry
    fm = variance_of_laplacian(img)
    if fm < threshold:
      text = "Blurry"
    rgb_img = BGR2RGB(img)
    cv2.putText(rgb_img, "{}: {:.2f}".format(text, fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    plt.imshow(rgb_img)
  plt.show()

  def num_of_edges_in_photo(image_path, lower_threshold, higher_threshold):
    # Load the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection algorithm
    edges = cv2.Canny(gray, lower_threshold, higher_threshold)

    # Count the number of edges
    num_edges = cv2.countNonZero(edges)

    # Print the number of edges
    return num_edges

  def ppi_resolution(image_path):
      from PIL import Image
      from fractions import Fraction

      # Load the image
      img = Image.open(image_path)

      # Extract the DPI information from the EXIF data
      dpi_x, dpi_y = img.info.get('dpi', (None, None))

      # Extract the physical size information from the EXIF data
      x_res, y_res = img.info.get('xresolution', None), img.info.get('yresolution', None)
      if x_res and y_res:
        x_res, y_res = Fraction(x_res[0], x_res[1]), Fraction(y_res[0], y_res[1])
        dpi_x, dpi_y = float(x_res), float(y_res)

      # Calculate the physical size of the image in inches
      if dpi_x and dpi_y:
        width_in = img.size[0] / dpi_x

      # Load the image in cv2
      img = cv2.imread(image_path)

      # Get the image size in pixels
      width_px, height_px = img.shape[:2]

      # Calculate the ppi/dpi of the image
      ppi = round(width_px / width_in)

      # return the results
      return ppi

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

 