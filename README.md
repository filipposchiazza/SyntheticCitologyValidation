# Synthetic Image Validation
This project is focused on validating the quality of synthetically generated images by comparing them with real images. The validation is based on the extraction of features from the images and the comparison of their distributions.
The idea behind this project is to provide a tool that can be used to assess the quality of synthetic images in the field of citology.

## Features Extraction
The feature extraction process is demonstrated in the Jupyter notebook **example_nuclei_features.ipynb**. This notebook shows how to extract features from cytology images. These features are extracted for a set of real and synthetically generated images and the distribution of the features is compared to assert if the synthetic images are similar to the real ones.

The features are extracted using the **extract_features** function from the **nuclei_features** module. The function takes an image and a mask as input and returns a set of features for each nucleus in the image.

## ECDF Validation
The validation of the synthetic images is based on the comparison of the Empirical Cumulative Distribution Function (ECDF) of the features extracted from the real and synthetic images. This process is demonstrated in the Jupyter notebook **example_ecdf_metric.ipynb**.

The ECDF comparison is performed using the ECDFDifference class from the **ecdf_validator** module. This class provides a method **evaluate_ecdf_differences** that takes the features of the real and synthetic images and returns the ECDF difference for each feature.

## Visualization
The project also provides functionality for visualizing the distribution of the features. This is done using the plot_features method of the HaralickValidator class from the **ecdf_validator** module. This method takes the features of the real and synthetic images and plots the distribution of each feature.

## Getting Started
To get started with the project, you can open the Jupyter notebooks **example_nuclei_features.ipynb** and **example_ecdf_metric.ipynb** and follow the examples provided there.

## Dependencies
The project requires the following dependencies:
- numpy 1.26.4
- matplotlib 3.8.4
- tqdm 4.66.4
- statsmodels 0.14.2
- mahotas 1.4.15
- cv2 4.10.0.84