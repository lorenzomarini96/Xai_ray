"""Xai_ray Project
XAI system for Pneumonia detection using chest X-ray images.

This python script evaluate implement a binary classification by means a
pretrained convolutional neural network (VGG16).
"""

import argparse
import os
from pathlib import Path
import sys
import math
import glob

# Packages for processing and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

# Package for ML and DL
import tensorflow as tf
from tensorflow import keras
#from keras.utils import load_img, img_to_array
from keras.utils.vis_utils import plot_model
from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.utils import to_categorical

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

# My helper functions
from helper_funcs import bar_plot, dataset_generation

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
        help="Data exploration: data set partition histogram and image visualization.",
        action='store_true'
    )

    parser.add_argument(
        "-e",
        "--epochs",
        metavar="",
        type=int,
        help="Number of epochs for train the CNN.",
        default="25",
    )

    args = parser.parse_args()

    #===========================================
    # STEP 1: Import data set, data exploration
    #===========================================
    
    # Path to the image dataset
    # PATH = '/Users/lorenzomarini/Desktop/Xai_ray/dataset/'
    PATH = args.datapath

    img_width = 196 # rendere opzionale come valore???
    img_height = 196 # rendere opzionale come valore???

    # Train directory
    train_folder=PATH+"train/"
    train_normal_dir=train_folder+"NORMAL/"
    train_pneu_dir=train_folder+"PNEUMONIA/"

    # Test directory
    test_folder=PATH+"test/"
    test_normal_dir=test_folder+"NORMAL/"
    test_pneu_dir=test_folder+"PNEUMONIA/"

    # Data exploration (statistics and bar plot)
    if args.dataexploration:
        bar_plot(PATH=PATH)

    #===========================================
    # STEP 2: Processing and training of CNN
    #===========================================
    #dataset_generation(PATH)

    #===========================================
    # STEP 3: Modeling
    #===========================================
    #model()


    """NUOVE PROVE"""
    # DATA GETHERING
    
    #-----------------------------------------
    # 1) listing the folders containing images

    # Train Dataset
    train_class_names=os.listdir(train_folder)
    #print(f"Train class names: {train_class_names}")

    # Test Dataset
    test_class_names=os.listdir(test_folder)
    #print(f"Test class names: {test_class_names}")

    #----------------------------------------------------
    # 2) Analysis of Train, Test and Validation directory
    
    # Read all files (with extension .jpeg)
    train_normal_cases = glob.glob(train_normal_dir + '*jpeg')
    train_pneu_cases = glob.glob(train_pneu_dir + '*jpeg')

    test_normal_cases = glob.glob(test_normal_dir + '*jpeg')
    test_pneu_cases = glob.glob(test_pneu_dir + '*jpeg')

    # create lists for train & test cases, create labels as well
    train_list = []
    test_list = []

    for x in train_normal_cases:
        train_list.append([x, "Normal"])
    
    for x in train_pneu_cases:
        train_list.append([x, "Pneumonia"])
    
    for x in test_normal_cases:
        test_list.append([x, "Normal"])
    
    for x in test_pneu_cases:
        test_list.append([x, "Pneumonia"])
    
    # create dataframes
    train_df = pd.DataFrame(train_list, columns=['image', 'Diagnos'])
    #print(train_df.shape)
    test_df = pd.DataFrame(test_list, columns=['image', 'Diagnos'])
    #print(test_df.shape)

    #--------------------------------------
    # PLOTTING RAW IMAGES (just for review)
    '''
    plt.figure(figsize=(20,8))
    for i,img_path in enumerate(train_df[train_df['Diagnos'] == "Pneumonia"][0:4]['image']):
        plt.subplot(2,4,i+1)
        plt.axis('off')
        img = plt.imread(img_path)
        plt.imshow(img, cmap='gray')
        plt.title('Pneumonia')
    
    for i,img_path in enumerate(train_df[train_df['Diagnos'] == "Normal"][0:4]['image']):
        plt.subplot(2,4,4+i+1)
        plt.axis('off')
        img = plt.imread(img_path)
        plt.imshow(img, cmap='gray')
        plt.title('Normal')
    plt.show()
    '''

    # PREPARING TRAINING IMAGE DATA

    # Preparing Training image data (image array and class name) for processing

    # Declaring variables
    x=[] # to store array value of the images
    y=[] # to store the labels of the images

    for folder in os.listdir(train_folder):
        if not folder.startswith('.'): # otherwise, problem with .DS_store file

            image_list=os.listdir(os.path.join(train_folder,folder))

            for img_name in tqdm(image_list, ascii=True, desc=f"train/{folder}"):
                # Loading images and converting to arrary
                img=tf.keras.utils.load_img(os.path.join(train_folder,folder,img_name),
                                                    target_size=(img_width,img_height)
                                                    )
        
                # Appending the arrarys
                x.append(img) # appending image array
                y.append(train_class_names.index(folder)) # appending class index to the array

    
    # Preparing validation images data (image array and class name) for processing

    # Declaring variables
    test_images=[]
    test_image_label=[] # to store the labels of the images

    for folder in os.listdir(test_folder):
        if not folder.startswith('.'): # otherwise, problem with .DS_store file

            image_list=os.listdir(os.path.join(test_folder,folder))

            for img_name in tqdm(image_list, ascii=True, desc=f"test/{folder}"):
                # Loading images and converting to arrary
                img=tf.keras.utils.load_img(os.path.join(test_folder,folder,img_name),
                                                    target_size=(img_width,img_height)
                                                    )
        
                # Appending arrays
                test_images.append(img) # appending image array
                test_image_label.append(test_class_names.index(folder))
    
    # Verifying the output

    # Training Dataset
    print("Training Dataset")

   #x=np.array(x) # Converting to np arrary to pass to the model
    #print(len(x))
    print(x.shape)

    #y=np.array(y)
    print(y)
    print(y.shape)

    # ===========

    # Test Dataset
    print("Test Dataset")

    #test_images=np.array(test_images) 
    print(test_images.shape)

    #test_image_label=np.array(test_image_label)
    print(test_image_label.shape)

    # ===========

