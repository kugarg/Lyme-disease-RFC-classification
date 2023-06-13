# Building a binary classification machine learning model: A guide to predicting participation in a Lyme disease program at a medical institute

This repository contains the code and data for constructing a Random Forest classification model to predict if individuals visiting Sanoviv Medical Institute in Mexico between 2020-2023 participated in the Lyme disease program based on their age, symptoms, blood count, and chemistry results. The model was built using Python and various libraries such as NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, and more.

The framework presented in this repository and the accompanying chapter in the book "Borrelia burgdorferi: Methods and Protocols," edited by Dr. Leona Gilbert, aims to provide a step-by-step guide for researchers to build machine-learning models effectively. By following this guide, researchers can gain insights into constructing and evaluating the Random Forest classification model.

## Table of Contents
1. [Introduction](#introduction)
2. [Data](#data)
3. [Folder Structure and Content](#folder-structure-and-content)
4. [Steps to Download and Reproduce](#steps-to-download-and-reproduce)
5. [Step-by-Step Framework](#step-by-step-framework)
    - [1. Import Libraries](#1-import-libraries)
    - [2. Import Data](#2-import-data)
    - [3. Exploratory Data Analysis](#3-exploratory-data-analysis)
    - [4. Divide Data into Training and Testing Sets](#4-divide-data-into-training-and-testing-sets)
    - [5. Feature Engineering](#5-feature-engineering)
    - [6. Feature Scaling](#6-feature-scaling)
    - [7. Feature Selection](#7-feature-selection)
    - [8. Model building, hyperparameter tuning, and k-fold cross validation](#8-model-building-hyperparameter-tuning-and-k-fold-cross-validation)
    - [9. Analyzing and Visualizing Model Performance](#9-analyzing-and-visualizing-model-performance)
6. [Results](#results)
7. [Feedback](#feedback)

## Introduction

The book chapter "Building a binary classification machine learning model: A guide to predicting participation in a Lyme disease program at a medical institute" aims to provide a comprehensive framework for building a Random Forest classification model using machine learning techniques. The model predicts if individuals visiting Sanoviv Medical Institute between 2020-2023 participated in the Lyme disease program based on various biomedical factors.

## Data

The data used in this study were retrospectively obtained from the Registry of Sanoviv Medical Institute patients from 2020 to 2023. The dataset includes information about 50 patients who attended the Lyme program and a control group of 72 individuals who participated in other medical programs. The data collection process adhered to the Declaration of Helsinki and Access to Information and Personal Data Protection guidelines provided by the National Institute of Transparency, Mexico.

## Folder Structure and Content

The repository has the following folder structure:
- **data**: Contains the `data.csv` file, which contains the dataset used for modeling.
- **figures**: Contains 11 figures that visualize missing data, exploratory data analysis, feature engineering results, and final model performance results.
- **notebook**: Contains `Random_Forest_Classification.ipynb` demonstrating the Python code used in the accompanying chapter.

## Steps to Download and Reproduce

To download this repository and reproduce the code and results, follow these steps:

1. Click on the green "<> Code".
2. Choose either "Download ZIP" to download the repository as a ZIP file or copy its URL to clone it using Git.
3. Extract the downloaded ZIP file (if applicable).
4. Follow the instructions in the "Getting Started" section above to set up the required environment and run the code.
5. Install the required Python libraries using pip from your command line, such as numpy, pandas, matplotlib, seaborn, scikit-plot, scipy, scikit-learn, and feature-engine.
6. Open the notebook file `Random_Forest_Classification.ipynb` in Google Colab or any compatible environment.
7. Execute the code cells sequentially to reproduce the analysis and model.

## Step-by-Step Framework

The framework for constructing the Random Forest classification model is divided into the following sections:

### 1. Import Libraries

This section imports the required data science libraries such as NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, and more.

### 2. Import Data

This section loads the dataset (`data.csv`) into a Pandas DataFrame for further analysis and modeling.

### 3. Exploratory Data Analysis

This section performs fundamental exploratory data analysis to gain insights into the dataset, including data summary, missing data visualization, and feature distributions.

### 4. Divide Data into Training and Testing Sets

The dataset is split into training and testing sets to train and evaluate the Random Forest classification model.

### 5. Feature Engineering

This section focuses on preprocessing the data by handling missing values, encoding categorical variables, and creating new features if required.

### 6. Feature Scaling

The features are scaled in this section to ensure all variables have a similar range, which can improve the model's performance.

### 7. Feature Selection

Here, feature selection techniques are employed to identify the most relevant features for the model.

### 8. Model building, hyperparameter tuning, and k-fold cross validation

This section uses the preprocessed data to build the Random Forest classification model. Hyperparameters of the Random Forest model are tuned to optimize its performance using RandomizedSearchCV. The model's performance is validated using k-fold cross-validation and evaluated based on accuracy, precision, recall, and F1 score metrics.

### 9. Analyzing and Visualizing Model Performance

Finally, the model's performance is analyzed and visualized using various plots, including the confusion matrix, ROC curve, and precision-recall curve.

## Results

The cross-validation results showed an average accuracy of 78.16% with a standard deviation of 12.15%. The ROC-AUC for the training set was 0.92, while the test set was 0.88. 

## Feedback

Feedback and suggestions are welcome. Please provide any feedback you may have regarding this project by emailing kunal.garg@tezted.com.
