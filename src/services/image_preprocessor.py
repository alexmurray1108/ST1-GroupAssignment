"""
*******************************
Assessment: 3
Author: u3271260
Group Members: u3281627, u3294093
Date: 30/04/2026
Group Assignment
*******************************
"""

import cv2
import numpy as np


class ImagePreprocessor:
    def __init__(self, image_size=(128, 128)):
        self.image_size = image_size

    def transform(self, file_path):
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Could not load image: {file_path}")

        resized = cv2.resize(image, self.image_size)
        normalized = resized.astype("float32") / 255.0
        flattened = normalized.flatten()

        return flattened