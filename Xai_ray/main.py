"""Xai_ray Project
XAI system for Pneumonia detection using chest X-ray images.

This python script evaluate implement a binary classification by means a
convolutional neural network pretrained (VGG16).
"""

import argparse
import os
from pathlib import Path
import sys
import math

# Packages for processing and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Package for ML and DL
import tensorflow as tf
from tensorflow import keras
from keras.utils.vis_utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

# My helper functions
from helper_funcs import bar_plot

sys.path.insert(0, str(Path(os.getcwd()).parent))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Pneumonia classification with explainable AI"
    )

    parser.add_argument(
        "-dp",
        "--datapath",
        metavar="",
        help="path of the data folder.",
        default="/Users/lorenzomarini/Desktop/Xai_ray/dataset/"
    )

    parser.add_argument(
        "-de",
        "--dataexploration",
        metavar="",
        type=bool,
        help="Data exploration: data set partition histogram and image visualization.",
        default=False,
    )

    args = parser.parse_args()

    #===========================================
    # STEP 1: Import data set, data exploration
    #===========================================
    
    # Path to the image dataset
    # PATH = '/Users/lorenzomarini/Desktop/Xai_ray/dataset/'
    PATH = args.datapath
    TRAIN_PATH = os.path.join(PATH, 'train')
    TEST_PATH = os.path.join(PATH, 'test')

    # Print the number of image in each subfolder
    print("\nTrain set:\n-----------------------")
    print(f"PNEUMONIA = {len(os.listdir(os.path.join(TRAIN_PATH, 'PNEUMONIA')))}")
    print(f"NORMAL = {len(os.listdir(os.path.join(TRAIN_PATH, 'NORMAL')))}")
    print("\nTest set:\n-----------------------")
    print(f"PNEUMONIA = {len(os.listdir(os.path.join(TEST_PATH, 'PNEUMONIA')))}")
    print(f"NORMAL = {len(os.listdir(os.path.join(TEST_PATH, 'NORMAL')))}")

    # Bar plot
    
    bar_plot(y_train, y_test)

