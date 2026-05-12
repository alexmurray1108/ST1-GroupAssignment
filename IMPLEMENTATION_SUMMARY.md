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

Our project was designed as a Image processing python application. We seperated Our stages into different modules and classes to ensure easier testing, maintenance and explanations. Our Stage One EDA generates charts and output summaries and Stage 2 trains the baseline classifier on preprocessed grayscale images. Our Stage 3 then allows a user to interact with the data through a menu-driven console.

"""Packages used"""
joblib
matplotlib
numpy
opencv-python
pandas
scikit-learn
seaborn

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