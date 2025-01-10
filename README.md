# Synthetic Image Validation
This project is focused on validating the quality of synthetically generated images with the following methods:
- Kernel Inception Distance 
- Unbiased Frechet Inception Distance
- Cell feature distribution comparison through Wasserstein distance
The idea behind this project is to provide a procedure that can be used to assess the quality of synthetic images in the field of citology.

## Repository Structure
The files are organized as follows:
- `dataset.py`: Contains the torch implementation of the dataset class.
- `metrics.py`: Contains the implementation of the Kernel Inception Distance and Unbiased Frechet Inception Distance.
- `metrics_analysis.ipynb`: Jupyter notebook that demonstrates how to analyze the metrics results.
- `cell_features.py`: Contains the implementation of the feature extraction process.
- `cell_features_analysis.ipynb`: Jupyter notebook that demonstrates how to analyze the features extracted from the images.
- `config.json`: Contains the configuration parameters for the project.


## Dependencies
The project requires the following dependencies:
- numpy 1.26.4
- matplotlib 3.8.4
- pandas 1.4.0
- tqdm 4.66.4
- statsmodels 0.14.2
- mahotas 1.4.15
- cv2 4.10.0.84
- torchmetrics 0.5.0