"""
***********************************
Assessment: 3
Author: u3271260
Group Members: u3281627, u3294093
Date: 28/04/2026
Group Assignment

This module preprocesses macroinvertebrate images for Stage 2
classification. It converts each raw image into a consistent numeric
feature vector that can be used by the classifier service.
***********************************
"""

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


class ImagePreprocessor:
    """Convert raw image files into normalized model-ready features."""

    def __init__(self, image_size: tuple[int, int] = (128, 128)) -> None:
        """Store the target image size used during preprocessing."""
        self.image_size = image_size

    def transform(self, file_path: str | Path) -> np.ndarray:
        """
        Load one image and convert it into a flattened numeric feature vector.

        Processing steps:
        1. Read the image in grayscale.
        2. Resize the image to the fixed project image size.
        3. Normalize pixel values from 0-255 to 0-1.
        4. Flatten the 2D image into a 1D vector for Scikit-learn.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {file_path}")

        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Could not read image file: {file_path}")

        resized = cv2.resize(image, self.image_size)
        normalized = resized.astype("float32") / 255.0
        flattened = normalized.flatten()

        return flattened

    def batch_transform(self, file_paths: Iterable[str | Path]) -> np.ndarray:
        """Preprocess multiple images and return them as a feature matrix."""
        features = []

        for file_path in file_paths:
            features.append(self.transform(file_path))

        return np.array(features)

    def transform_for_display(self, file_path: str | Path) -> np.ndarray:
        """
        Preprocess one image but keep it in 2D form for checking or display.

        This method is useful for testing because it shows the resized and
        normalized image before it is flattened for the classifier.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {file_path}")

        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Could not read image file: {file_path}")

        resized = cv2.resize(image, self.image_size)
        normalized = resized.astype("float32") / 255.0

        return normalized


if __name__ == "__main__":
    print("--- Image Preprocessor Test ---")

    preprocessor = ImagePreprocessor(image_size=(128, 128))
    data_dir = Path("../../data/raw")
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    sample_images = [
        path for path in data_dir.rglob("*")
        if path.suffix.lower() in supported_extensions
    ]

    if not sample_images:
        print("No sample images found in data/raw.")
        print("Place the Kaggle dataset inside data/raw before testing.")
    else:
        sample_path = sample_images[0]
        features = preprocessor.transform(sample_path)
        display_image = preprocessor.transform_for_display(sample_path)

        print(f"Sample image: {sample_path}")
        print(f"Flattened feature shape: {features.shape}")
        print(f"Display image shape: {display_image.shape}")
        print(f"Minimum pixel value: {features.min():.4f}")
        print(f"Maximum pixel value: {features.max():.4f}")
        print("Preprocessing test completed successfully.")

    print(" Test Finished ")

