"""Helper functions to Xai_ray prject."""

import os
import glob
import logging
import itertools

import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def bar_plot(y_train, y_test):
    """Plot an horizontal bar graph
    showing the relative partition of the two given dataset.

    Parameters
    ----------
    y_train : numpy_array
        Labels for the train data set.
    y_test : numpy_array
        Labels for the train data set.

    Returns
    -------
    None

    Examples
    --------
    >>> import numpy as np
    >>> y_train = np.array([1, 0, 0, 1, 1])
    >>> y_test = np.array([1, 1, 0, 1, 1])
    >>> bar_plot(y_train, y_test)
    """



def read_imgs(dataset_path, classes):
    """Function reading all the images in a given folder which already contains
    two subfolder.

    Parameters
    ----------
        dataset_path : str
            Path to the image folder.
        classes : list
            0 and 1 mean normal tissue and microcalcification clusters, respectively.

    Returns
    -------
        array: numpy_array
            Array containing the value of image/label.

    Examples
    --------
    >>> TRAIN_DATA_PATH = '/path/to/train/folder'
    >>> x_train, y_train = read_imgs(TRAIN_DATA_PATH, [0, 1])
    """
    tmp = []
    labels = []
    for cls in classes:
        try:
            fnames = glob.glob(os.path.join(dataset_path, str(cls), '*.jpeg'))
            logging.info(f'Read images from class {cls}')
            tmp += [imread(fname) for fname in fnames]
            labels += len(fnames)*[cls]
        except Exception as e_error:
            raise Exception('Image or path not found') from e_error
    logging.info(f'Loaded images from {dataset_path}')

    return np.array(tmp, dtype='float32')[..., np.newaxis]/255, np.array(labels)
