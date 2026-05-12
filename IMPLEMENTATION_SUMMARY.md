"""
*******************************
Assessment: 3
Author: u3281627
Group Members: u3294093, u3271260
Date: 28/04/2026 (TODAYS DATE)
Group Assignment


NOTE: Code is adapted from the Assignment 3 Full Guidance,
with some modifications to better fit the needs of this project.
*******************************
"""

# Macroinvertebrate Image Analysis System
# u3281627, u3294093 & u3271260 
The goal is to build an automated system for identifying aquatic macroinvertebrate species from image data.

"""This was taken straight from canvas, please paraphrase v
"""
Our project was designed as a modular Python application. We separated
dataset indexing, exploratory data analysis, image preprocessing,
classification, and user interaction into different classes and modules.
This made the project easier to test, maintain, and explain. The Stage 1
workflow generates charts and summary outputs from the dataset, while the
Stage 2 workflow trains a baseline classifier on preprocessed grayscale
images. The deployed application then loads the trained model and allows a
user to select or enter an image and view a predicted macroinvertebrate
class.

"""Insert Requirements.txt with an explanation"""

"""Key features Implemented"""
- Dataset indexing from `data/raw/`
- Class distribution analysis
- Image width and height analysis
- Sample image grid generation
- Image preprocessing into flattened feature vectors
- Baseline classifier training with evaluation metrics
- Console menu interface for running the workflow

"""Testing summary"""
- Invalid Image Path
- Invalid Menu Option
- Valid Prediction
- Misreading Image Path

Please find further explanation in 'Manual_Testing.md'

"""Sample Outputs"""
Macoinvertebrate Image Analysis & Processing System
1. Show dataset summary
2. Generate EDA outputs
3. Train the baseline classifier
4. Predictive analysis of an image
5. Exit
Please select an option: 1
{'total_images': 2665, 'total_classes': 17, 'mean_width': 599.2660412757974, 'mean_height': 389.98198874296435}

Macoinvertebrate Image Analysis & Processing System
1. Show dataset summary
2. Generate EDA outputs
3. Train the baseline classifier
4. Predictive analysis of an image
5. Exit
Please select an option: 5
Exiting application

"""Adapted Code acknowledgement"""
Code is adapted from the Assignment 3 Full Guidance,
with some modifications to better fit the needs of this project.

"""Work Division""
u3281627 - Stage One: EDA, dataset Analysis, workflow integration, main, Implementation Summary & GUI
u3271260 - Image preprocessing, preprocessing testing, presentation slides
u3294093 - Machine learning classification model, model training and testing, major editing and testing of the program.