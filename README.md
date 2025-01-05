# XAI System for Pneumonia Detection Using Chest X-ray Images

**Authors**: Alessia Fantini, Lorenzo Marini, Alessandro Quarta

## Overview

This project focuses on building an Explainable AI (XAI) system for pneumonia detection using chest X-ray images. The system leverages machine learning models to classify images into two categories: Pneumonia and Normal, providing explanations for its predictions to aid medical professionals.

## Dataset

The dataset used in this project is available on [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). It consists of 5,863 chest X-ray images, categorized into two groups: Pneumonia and Normal. The images are organized into three main folders:

- **train**: 3,563 images for model training
- **test**: 1,000 images for model testing
- **val**: 1,000 images for model validation

The X-ray images come from a retrospective cohort of pediatric patients (ages 1 to 5) at the Guangzhou Women and Children’s Medical Center, Guangzhou. The images were pre-screened for quality and graded by expert physicians to ensure high accuracy and reliability in the dataset.

## Task to Solve

The goal of this project is to create a machine learning pipeline for pneumonia detection in chest X-rays, employing a classification model to categorize the images into two classes: Pneumonia or Normal.

Additionally, the project aims to enhance the model with Explainable AI techniques to provide transparency in its decision-making process, which is crucial in the medical field for building trust in AI-assisted diagnoses.

## Machine Learning Pipeline

1. **Feature Construction**: 
   - Preprocessing of the X-ray images (resizing, normalization).
   - Feature extraction from images (e.g., pixel intensity, texture features).

2. **Model Construction**: 
   - Model training using deep learning techniques, including convolutional neural networks (CNNs).
   - Hyperparameter tuning for model optimization.

3. **Model Validation**: 
   - Cross-validation techniques to assess model generalization.
   - Evaluation metrics: accuracy, precision, recall, F1-score.

4. **Interpretation**: 
   - Using Explainable AI (XAI) techniques such as Grad-CAM to visualize which parts of the image contribute most to the model's decision.
   - Understanding and interpreting the model's behavior for better usability in clinical settings.

5. **Performance Evaluation**:
   - Classification Report (accuracy, precision, recall, F1-score).
   - Confusion Matrix to visualize true positives, false positives, true negatives, and false negatives.
   - ROC Curve to evaluate the model's performance at different classification thresholds.

## XAI Methodology

The XAI component of this project is designed to help users understand how the model makes predictions. Using techniques like Grad-CAM, we can visualize which regions of the chest X-ray images are most influential in determining whether the image is classified as "Pneumonia" or "Normal". This approach improves model transparency and allows clinicians to trust and interpret AI results better.

## Conclusion

This project demonstrates the potential of using machine learning and Explainable AI for medical image analysis. The system not only provides accurate predictions for pneumonia detection but also offers interpretable insights into the reasoning behind each decision, which is crucial in medical applications.

## References

- [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
