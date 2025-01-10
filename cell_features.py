import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import os
from mahotas.features import haralick
import pandas as pd


FEATURES_NAMES = ['Centroid X',
                  'Centroid Y',
                  'Area',
                  'Perimeter',
                  'Equivalent Diameter', 
                  'Circularity',
                  'Convexity',
                  'Aspect Ratio',
                  'Minor Axis',
                  'Major Axis',
                  'Eccentricity',
                  'Extent', 
                  'Avg_red',
                  'Min_red',
                  'Max_red',
                  'Std_red', 
                  'Avg_green',
                  'Min_green',
                  'Max_green',
                  'Std_green', 
                  'Avg_blue',
                  'Min_blue',
                  'Max_blue',
                  'Std_blue',
                  'Avg_hue',
                  'Min_hue',
                  'Max_hue',
                  'Std_hue',
                  'Avg_saturation',
                  'Min_saturation',
                  'Max_saturation',
                  'Std_saturation',
                  'Avg_value',
                  'Min_value',
                  'Max_value',
                  'Std_value',
                  'Angular Second Moment',
                  'Contrast',
                  'Correlation',
                  'Variance',
                  'Inverse Difference Moment',
                  'Sum Average',
                  'Sum Variance',
                  'Sum Entropy',
                  'Entropy',
                  'Difference Variance',
                  'Difference Entropy',
                  'Information Measure of Correlation 1',
                  'Information Measure of Correlation 2']

def extract_features_distribution(directory, num_samples, seed):
    """ Extract the features from the images in the directory

    Parameters
    ----------
    directory : str
        The directory of the images
    num_samples : int
        The number of samples to evaluate
    seed : int
        The seed for the random shuffle

    Returns
    -------
    features : numpy array
        The features of the cells
    """

    # Get the image filenames in the directory and shuffle them
    img_filenames = [os.path.join(directory, img) for img in os.listdir(directory)]
    np.random.seed(seed)
    np.random.shuffle(img_filenames)

    # Evaluate the features for each image
    features = []
    problematic_cells = 0
    for img_file in tqdm(img_filenames[:num_samples]):
        data = plt.imread(img_file)
        img = data[:, :256, :3]
        mask = np.mean(data[:, 256:, :3], axis=-1).astype(np.uint8)
        img_features, prob = extract_features_from_patch(img, mask)
        problematic_cells += prob
        if img_features.shape[0] > 0:
            features.append(img_features)
    features = np.concatenate(features, axis=0)
    total_num_nuclei = features.shape[0]

    print(f'Number of succesfully processed cells: {total_num_nuclei}\nNumber of problematic cells: {problematic_cells}')
    return features



def extract_features_from_patch(img, mask):
    """ Extract features from the cells

    Parameters
    ----------
    img : numpy array
        The original cythological image
    mask : numpy array
        The mask of the cells
    
    Returns
    -------
    features : numpy array
        The cells features
    problematic_cells : int
        The number of problematic cells that could not be processed
    """

    # Get a mask for each cell
    cells_masks, cells_contours = mask_splitting(mask)

    # Get an image for each cell with the background removed
    cells_images = cells_splitting(img, cells_masks)

    # Check if the number of masks and images are the same
    assert len(cells_masks) == len(cells_images)

    # List to store the features
    features = []

    # Iterate over each cell
    problematic_cells = 0
    for i in range(len(cells_masks)):

        try:
            # Get the morphological features of the cell
            morph_features = get_morphological_features(cells_contours[i])

            # Check if morph features has some NaN values
            if np.any(np.isnan(morph_features)):
                problematic_cells += 1
                continue

            # Get the RGB color features of the cell
            color_features_RGB = get_RGB_color_features(cells_images[i], cells_masks[i])

            # Get the HSV color features of the cell
            color_features_HSV = get_HSV_color_features(cells_images[i], cells_masks[i])

            # Get the Haralick features of the cell
            haralick_features = get_haralick_features(cells_images[i])

            # Concatenate the features
            cell_features = np.array([*morph_features, *color_features_RGB, *color_features_HSV, *haralick_features])

            # Append the features to the list
            features.append(cell_features)
        except:
            problematic_cells += 1

    return np.asarray(features), problematic_cells



def mask_splitting(mask):
    """ Split the mask into individual masks for each cell 
    
    Parameters
    ----------
    mask : numpy array
        The mask of the cells
    
    Returns
    -------
    cells_masks : list
        List of individual masks for each cell
    contours : list
        List of contours for each cell
    """
    # Find contours
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # List to store individual masks
    cells_masks = []

    # Iterate over each contour and create a mask for it
    for i, contour in enumerate(contours):
        # Create an empty mask
        single_mask = np.zeros_like(mask)
        # Draw the contour on the mask
        cv2.drawContours(single_mask, [contour], -1, color=1, thickness=cv2.FILLED)
        # Append the mask to the list
        cells_masks.append(single_mask)

    return cells_masks, contours



def cells_splitting(img, cells_masks):
    """ Split the image into individual images for each cell

    Parameters
    ----------
    img : numpy array
        The image of the cells
    cells_masks : list
        List of individual masks for each cell
    
    Returns
    -------
    cells_images : list
        List of individual images for each cell
    """
    cells_images = []

    for cell_mask in cells_masks:
        cells_images.append(img * cell_mask[:, :, None])

    return cells_images


############################################################################################################
## CELL FEATURES
def get_morphological_features(contour):
    """ Get the morphological features of the cell

    Parameters
    ----------
    contour : numpy array
        The cell's contour, with shape (n, 1, 2)

    Returns
    -------
    cX : int
        The x-coordinate of the centroid
    cY : int
        The y-coordinate of the centroid
    area : float
        The cell's area
    perimeter : float
        The cell's perimeter
    eq_diameter : float
        The cell's equivalent diameter
    circularity : float
        The cell's circularity
    convexity : float
        The cell's convexity
    aspect_ratio : float
        The cell's aspect ratio
    extent : float
        The cell's extent
    """
    # Centroids
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    # Get the area of the contour
    area = cv2.contourArea(contour)

    # Get the perimeter of the contour
    perimeter = cv2.arcLength(curve=contour, closed=True)

    # Get the equivalent diameter
    eq_diameter = np.sqrt(4 * area / np.pi)

    # Get the circularity
    circularity = 4 * np.pi * (area / np.square(perimeter))

    # Get the convexity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(curve=hull, closed=True)
    convexity = area / hull_area

    # Get the aspect ratio
    _, _, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h

    # Get major, minor axis and eccentricity
    (_, _), (ma, MA), angle = cv2.fitEllipse(contour)
    eccentricity = np.sqrt(1 - (ma / MA) ** 2)

    # Get the extent
    rect_area = w * h
    extent = area / rect_area


    return cX, cY, area, perimeter, eq_diameter, circularity, convexity, aspect_ratio, ma, MA, eccentricity, extent



def get_RGB_color_features(img, mask):
    """ Get the average color of the cell

    Parameters
    ----------
    img : numpy array
        The image of the cell
    mask : numpy array
        The mask of the cell
    
    Returns
    -------
    average_red : float
        The average red color of the cell
    min_red : float
        The minimum red color of the cell
    max_red : float
        The maximum red color of the cell
    std_red : float
        The standard deviation of the red color of the cell
    average_green : float
        The average green color of the cell
    min_green : float
        The minimum green color of the cell
    max_green : float
        The maximum green color of the cell
    std_green : float
        The standard deviation of the green color of the cell
    average_blue : float
        The average blue color of the cell
    min_blue : float
        The minimum blue color of the cell
    max_blue : float
        The maximum blue color of the cell
    std_blue : float
        The standard deviation of the blue color of the cell
    """

    # Get the indices of the mask
    indices = np.where(mask == 1)
    
    # Get the average color of the cell
    average_red = np.mean(img[indices[0], indices[1], 0])
    min_red = np.min(img[indices[0], indices[1], 0])
    max_red = np.max(img[indices[0], indices[1], 0])
    std_red = np.std(img[indices[0], indices[1], 0])

    average_green = np.mean(img[indices[0], indices[1], 1])
    min_green = np.min(img[indices[0], indices[1], 1])
    max_green = np.max(img[indices[0], indices[1], 1])
    std_green = np.std(img[indices[0], indices[1], 1])

    average_blue = np.mean(img[indices[0], indices[1], 2])
    min_blue = np.min(img[indices[0], indices[1], 2])
    max_blue = np.max(img[indices[0], indices[1], 2])
    std_blue = np.std(img[indices[0], indices[1], 2])

    return average_red, min_red, max_red, std_red, average_green, min_green, max_green, std_green, average_blue, min_blue, max_blue, std_blue



def get_HSV_color_features(img, mask):
    """ Get the average color of the cell in the HSV color space

    Parameters
    ----------
    img : numpy array
        The image of the cell
    mask : numpy array
        The mask of the cell
    
    Returns
    -------
    average_hue : float
        The average hue of the cell
    min_hue : float
        The minimum hue of the cell
    max_hue : float
        The maximum hue of the cell
    std_hue : float
        The standard deviation of the hue of the cell
    average_saturation : float
        The average saturation of the cell
    min_saturation : float
        The minimum saturation of the cell
    max_saturation : float
        The maximum saturation of the cell
    std_saturation : float
        The standard deviation of the saturation of the cell
    average_value : float
        The average value of the cell
    min_value : float
        The minimum value of the cell
    max_value : float
        The maximum value of the cell
    std_value : float
        The standard deviation of the value of the cell
    """

    # Convert the image to the HSV color space
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Get the indices of the mask
    indices = np.where(mask == 1)
    
    # Get the average color of the cell
    average_hue = np.mean(img_hsv[indices[0], indices[1], 0])
    min_hue = np.min(img_hsv[indices[0], indices[1], 0])
    max_hue = np.max(img_hsv[indices[0], indices[1], 0])
    std_hue = np.std(img_hsv[indices[0], indices[1], 0])

    average_saturation = np.mean(img_hsv[indices[0], indices[1], 1])
    min_saturation = np.min(img_hsv[indices[0], indices[1], 1])
    max_saturation = np.max(img_hsv[indices[0], indices[1], 1])
    std_saturation = np.std(img_hsv[indices[0], indices[1], 1])

    average_value = np.mean(img_hsv[indices[0], indices[1], 2])
    min_value = np.min(img_hsv[indices[0], indices[1], 2])
    max_value = np.max(img_hsv[indices[0], indices[1], 2])
    std_value = np.std(img_hsv[indices[0], indices[1], 2])

    return average_hue, min_hue, max_hue, std_hue, average_saturation, min_saturation, max_saturation, std_saturation, average_value, min_value, max_value, std_value



def get_haralick_features(img):
    """ Get the Haralick features of the cell

    Parameters
    ----------
    img : numpy array
        The image of the cell
    
    Returns
    -------
    texture_features : numpy array
        The Haralick features of the cell
    """

    # Convert the img to the range 0-255 and uint8
    img = (img * 255).astype(np.uint8)

    # Convert the image to grayscale 
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Get the texture features only for the non-zero pixels
    texture_features = haralick(gray_img, ignore_zeros=True, return_mean=True)
    
    return texture_features



if __name__ == '__main__':

    with open('config.json', 'r') as f:
        config = json.load(f)

    # Setup directories and filenames
    img_dir = config['cell_features']['img_dir']
    num_samples_for_features = config['cell_features']['num_samples_for_features']
    seed = config['cell_features']['seed']
    save_filename = config['cell_features']['save_filename']

    # Extract features from the images
    features = extract_features_distribution(img_dir, num_samples_for_features, seed)

    # Save as csv
    df = pd.DataFrame(features, columns=FEATURES_NAMES)
    df.to_csv(save_filename, index=False)
