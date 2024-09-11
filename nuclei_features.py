import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import os
from mahotas.features import haralick

FEATURES_NAMES = ['Area', 
                  'Circularity', 
                  'Avg_red', 
                  'Avg_green', 
                  'Avg_blue',
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


def extract_features(img, mask):
    """ Extract features from the nuclei

    Parameters
    ----------
    img : numpy array
        The original cythological image
    mask : numpy array
        The mask of the nuclei
    
    Returns
    -------
    features : numpy array
        The nuclei features
    """

    # Get a mask for each nucleus
    nuclei_masks, nuclei_contours = mask_splitting(mask)

    # Get an image for each nucleus with the background removed
    nuclei_images = nuclei_splitting(img, nuclei_masks)

    # Check if the number of masks and images are the same
    assert len(nuclei_masks) == len(nuclei_images)

    # List to store the features
    features = []

    # Iterate over each nucleus
    for i in range(len(nuclei_masks)):

        try:
            # Get the area of the nucleus
            area = get_nucleus_area(nuclei_masks[i])

            # Get the circularity of the nucleus
            circularity = get_nucleus_circularity(nuclei_contours[i])

            # Get the average color of the nucleus
            average_color = get_average_color(nuclei_images[i], nuclei_masks[i])

            # Get the Haralick features of the nucleus
            haralick_features = get_haralick_features(nuclei_images[i])

            # Concatenate the features
            nucleus_features = np.array([area, circularity, *average_color, *haralick_features])

            # Append the features to the list
            features.append(nucleus_features)
        except:
            features.append(np.zeros(18))

    return np.asarray(features)



def mask_splitting(mask):
    """ Split the mask into individual masks for each nucleus 
    
    Parameters
    ----------
    mask : numpy array
        The mask of the nuclei
    
    Returns
    -------
    nuclei_masks : list
        List of individual masks for each nucleus
    contours : list
        List of contours for each nucleus
    """
    # Find contours
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # List to store individual masks
    nuclei_masks = []

    # Iterate over each contour and create a mask for it
    for i, contour in enumerate(contours):
        # Create an empty mask
        single_mask = np.zeros_like(mask)
        # Draw the contour on the mask
        cv2.drawContours(single_mask, [contour], -1, color=1, thickness=cv2.FILLED)
        # Append the mask to the list
        nuclei_masks.append(single_mask)

    return nuclei_masks, contours



def nuclei_splitting(img, nuclei_masks):
    """ Split the image into individual images for each nucleus

    Parameters
    ----------
    img : numpy array
        The image of the nuclei
    nuclei_masks : list
        List of individual masks for each nucleus
    
    Returns
    -------
    nuclei_images : list
        List of individual images for each nucleus
    """
    nuclei_images = []

    for nucleus_mask in nuclei_masks:
        nuclei_images.append(img * nucleus_mask[:, :, None])

    return nuclei_images



def get_nucleus_area(mask):
    """ Get the area of the nucleus

    Parameters
    ----------
    mask : numpy array
        The mask of the nucleus
    
    Returns
    -------
    area : float
        The area of the nucleus
    """
    return np.sum(mask) / mask.size



def get_nucleus_circularity(contour):
    """ Get the circularity of the nucleus

    Parameters
    ----------
    contour : numpy array
        The contour of the nucleus  
    
    Returns
    -------
    circularity : float
        The circularity of the nucleus
    """

    # Compute the area of the contour
    area = cv2.contourArea(contour)
    
    # Compute the perimeter of the contour
    perimeter = cv2.arcLength(contour, True)
    
    if perimeter == 0:
        return 0
    
    # Compute circularity
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    
    return circularity



def get_average_color(img, mask):
    """ Get the average color of the nucleus

    Parameters
    ----------
    img : numpy array
        The image of the nucleus
    mask : numpy array
        The mask of the nucleus
    
    Returns
    -------
    average_red : float
        The average red value of the nucleus
    average_green : float
        The average green value of the nucleus
    average_blue : float
        The average blue value of the nucleus
    """

    # Get the indices of the mask
    indices = np.where(mask == 1)
    
    # Get the average color of the nucleus
    average_red = np.mean(img[indices[0], indices[1], 0])
    average_green = np.mean(img[indices[0], indices[1], 1])
    average_blue = np.mean(img[indices[0], indices[1], 2])
    
    return average_red, average_green, average_blue



def get_haralick_features(img):
    """ Get the Haralick features of the nucleus

    Parameters
    ----------
    img : numpy array
        The image of the nucleus
    
    Returns
    -------
    texture_features : numpy array
        The Haralick features of the nucleus
    """

    # Convert the img to the range 0-255 and uint8
    img = (img * 255).astype(np.uint8)

    # Convert the image to grayscale 
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Get the texture features only for the non-zero pixels
    texture_features = haralick(gray_img, ignore_zeros=True).mean(axis=0)
    
    return texture_features