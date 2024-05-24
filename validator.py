from statsmodels.distributions.empirical_distribution import ECDF
import mahotas 
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import random
import os



def ECDFDifference(data1, data2, bins=10000):
    """Calculate the difference between two empirical cumulative distribution functions (ECDFs) of two datasets.

    Parameters
    ----------
    data1 : array_like
        First dataset.
    data2 : array_like
        Second dataset.
    bins : int, optional
        Number of bins to use for the ECDF. The default is 10000.

    Returns
    -------
    ecdf_diff : float
        The difference between the two ECDFs.
    """
    ecdf1 = ECDF(data1)
    ecdf2 = ECDF(data2)
    n1 = len(data1)
    n2 = len(data2)

    combined_data = np.sort(np.concatenate([data1, data2]))
    range_min = np.min(combined_data)
    range_max = np.max(combined_data)
    bin_edges = np.linspace(range_min, range_max, bins)
    dy = (ecdf1(bin_edges) - ecdf2(bin_edges))[:-1]
    dx = np.diff(bin_edges)
    ecdf_diff = np.sum(np.abs(dy) * dx)
    return ecdf_diff




class HaralickValidator:

    def __init__(self, img_dir1, img_dir2, num_samples):
        """Initialize the HaralickValidator object.
        
        Parameters
        ----------
        img_dir1 : str
            Path to the directory containing the first dataset of images.
        img_dir2 : str
            Path to the directory containing the second dataset of images.
        num_samples : int
            Number of samples to use from each dataset.
        """
        self.img_dir1 = img_dir1
        self.img_dir2 = img_dir2
        self.num_samples = num_samples
        self.features_names = ['Angular Second Moment', 
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
        
        self.filenames1, self.filenames2 = self.check_directories()
        


    def check_directories(self):
        """Check if the directories exist and have enough images.

        Returns
        -------
        filenames1 : list
            List of filenames in the first directory.
        filenames2 : list
            List of filenames in the second directory.
        """
        if not os.path.exists(self.img_dir1):
            raise ValueError(f"Directory '{self.img_dir1}' does not exist.")
        if not os.path.exists(self.img_dir2):
            raise ValueError(f"Directory '{self.img_dir2}' does not exist.")
        
        filenames1 = os.listdir(self.img_dir1)
        filenames2 = os.listdir(self.img_dir2)
        if len(filenames1) < self.num_samples or len(filenames2) < self.num_samples:
            raise ValueError(f"Number of images in directories is less than the number of samples.")
        
        filenames1  = random.sample(filenames1, self.num_samples)
        filenames2  = random.sample(filenames2, self.num_samples)
        
        return filenames1, filenames2




    def extract_features_from_image(self, img):
        """Extract Haralick texture features from an image.

        Parameters
        ----------
        img : numpy.ndarray
            Image to extract features from.
        
        Returns
        -------
        ht_mean : numpy.ndarray
            Mean of the Haralick texture features.
        """
        # calculate haralick texture features for 4 types of adjacency
        textures = mahotas.features.haralick(img)
        # take the mean of it and return it
        ht_mean = textures.mean(axis=0)
        return ht_mean
    


    def extract_features(self):
        """Extract Haralick texture features from the images in the directories.

        Returns
        -------
        features1 : numpy.ndarray
            Features extracted from the first dataset.
        features2 : numpy.ndarray
            Features extracted from the second dataset.
        """

        features1 = []
        features2 = []

        for (filename1, filename2) in tqdm(zip(self.filenames1, self.filenames2)):
            img1 = cv2.imread(os.path.join(self.img_dir1, filename1), cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(os.path.join(self.img_dir2, filename2), cv2.IMREAD_GRAYSCALE)
            features1.append(self.extract_features_from_image(img1))
            features2.append(self.extract_features_from_image(img2))

        return np.array(features1), np.array(features2)
    


    def plot_features(self, features1, features2, figsize=(20, 20), label1 = 'dataset1', label2 = 'dataset2', bins=100, density=True):
        """Plot the histograms of the Haralick texture features.

        Parameters
        ----------
        features1 : numpy.ndarray
            Features extracted from the first dataset.
        features2 : numpy.ndarray
            Features extracted from the second dataset.
        figsize : tuple, optional
            Size of the figure. The default is (20, 20).
        label1 : str, optional
            Label for the first dataset. The default is 'dataset1'.
        label2 : str, optional
            Label for the second dataset. The default is 'dataset2'.
        bins : int, optional
            Number of bins for the histograms. The default is 100.
        density : bool, optional
            If True, the histogram is normalized. The default is True.
        """
        fig, axs = plt.subplots(4, 4, figsize=figsize)
        axs[3, 1].axis('off')
        axs[3, 2].axis('off')
        axs[3, 3].axis('off')

        for i in range(13):
            r = i // 4
            c = i - 4 * r
            ax = axs[r, c]
            ax.hist(features1[:, i], bins=bins, alpha=0.5, label=label1, color='blue', density=density)
            ax.hist(features2[:, i], bins=bins, alpha=0.5, label=label2, color='orange', density=density)
            ax.set_title(self.features_names[i])
            ax.legend(loc='upper right')

    

    def evaluate_ecdf_differences(self, features1, features2, **kwargs):
        """Evaluate the differences between the ECDFs of the Haralick texture features.

        Parameters
        ----------
        features1 : numpy.ndarray
            Features extracted from the first dataset.
        features2 : numpy.ndarray
            Features extracted from the second dataset.
        **kwargs : dict
            Additional arguments to pass to the ECDF
        """
        ecdf_diff = {}
        for i in range(13):
            ecdf_diff[self.features_names[i]] = ECDFDifference(features1[:, i], features2[:, i], bins=kwargs.get('bins', 10000))

        return ecdf_diff
    


    def plot_ecdf(self, features1, features2, figsize=(20, 20), label1='dataset1', label2='dataset2', **kwargs):
        """Plot the ECDFs of the Haralick texture features.

        Parameters
        ----------
        features1 : numpy.ndarray
            Features extracted from the first dataset.
        features2 : numpy.ndarray
            Features extracted from the second dataset.
        figsize : tuple, optional
            Size of the figure. The default is (20, 20).
        label1 : str, optional
            Label for the first dataset. The default is 'dataset1'.
        label2 : str, optional
            Label for the second dataset. The default is 'dataset2'.
        **kwargs : dict
            Additional arguments to pass to the ECDF
        """

        fig, axs = plt.subplots(4, 4, figsize=figsize)
        axs[3, 1].axis('off')
        axs[3, 2].axis('off')
        axs[3, 3].axis('off')

        for i in range(13):
            r = i // 4
            c = i - 4 * r
            ax = axs[r, c]
            ecdf1 = ECDF(features1[:, i])
            ecdf2 = ECDF(features2[:, i])
            combined_features = np.concatenate([features1[:, i], features2[:, i]])
            range_min = np.min(combined_features)
            range_max = np.max(combined_features)
            x = np.linspace(range_min, range_max, kwargs.get('bins', 10000))
            ax.plot(x, ecdf1(x), label=label1, color='blue')
            ax.plot(x, ecdf2(x), label=label2, color='orange')
            ax.plot(x, np.abs(ecdf1(x) - ecdf2(x)), label='Difference', color='red')
            ax.set_title(self.features_names[i])
            ax.legend(loc='best')
       