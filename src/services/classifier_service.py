"""
***********************************
Assessment: 3
Author: u3294093
Group Members: u3281627, u3271260
Date: 28/04/2024
Group Assignment

This module implements a small 
transfer-learning wrapper around 
TensorFlow / Keras.

NOTE: This model requires Python 
3.12 or older to match the 
project's TensorFlow compatibility.
***********************************
"""

from pathlib import Path

import tensorflow as tf
from tensorflow import keras


class TransferLearningService:
    """
    Wrap a transfer-learning workflow for an optional advanced classifier.
    """
    def __init__(
        self,
        image_size: tuple[int, int] = (224, 224),
        batch_size: int = 32
    ) -> None:
        print(f"[INIT] Initializing TransferLearningService with image_size={image_size}, batch_size={batch_size}")
        self.image_size = image_size
        self.batch_size = batch_size
        self.model = None
        print("[INIT] TransferLearningService initialized successfully")

    def build_datasets(
        self,
        train_dir: Path | str,
        validation_split: float = 0.2,
    ):
        """
        Create training and validation datasets from a class-folder structure.
        """
        print(f"[BUILD_DATASETS] Loading datasets from {train_dir} with validation_split={validation_split}")
        train_ds = keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=validation_split,
            subset="training",
            seed=42,
            image_size=self.image_size,
            batch_size=self.batch_size
        )
        print(f"[BUILD_DATASETS] Training dataset created successfully")

        val_ds = keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=validation_split,
            subset="validation",
            seed=42,
            image_size=self.image_size,
            batch_size=self.batch_size
        )
        print(f"[BUILD_DATASETS] Validation dataset created successfully")

        return train_ds, val_ds

    def build_model(self, class_count: int) -> keras.Model:
        """
        Build and compile the transfer-learning model.
        """
        print(f"[BUILD_MODEL] Building transfer learning model for {class_count} classes")
        print(f"[BUILD_MODEL] Loading MobileNetV2 base model with ImageNet weights...")
        base_model = keras.applications.MobileNetV2(
            input_shape=self.image_size + (3,),
            include_top=False,
            weights="imagenet"
        )
        print(f"[BUILD_MODEL] Base model loaded successfully")

        base_model.trainable = False
        print(f"[BUILD_MODEL] Base model layers frozen")

        self.model = keras.Sequential([
            keras.layers.Rescaling(1.0 / 255),
            base_model,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(class_count, activation="softmax"),
        ])
        print(f"[BUILD_MODEL] Sequential model architecture created")

        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        print(f"[BUILD_MODEL] Model compiled successfully")

        return self.model

"""
NOTE: Code is adapted from the Assignment 3 Full Guidance, 
with some modifications to better fit the needs of this project.
"""

if __name__ == "__main__":
    print("\n--- Classifier Service Test ---\n")
    
    # Initialize the service
    service = TransferLearningService(image_size=(224, 224), batch_size=32)
    
    # Build model with 16 classes
    print("\nBuilding model...")
    model = service.build_model(class_count=16)
    
    # Try to load datasets if data exists
    print("\nTesting dataset loading...")
    data_dir = Path("data/raw")
    if data_dir.exists():
        try:
            train_ds, val_ds = service.build_datasets(data_dir, validation_split=0.2)
        except Exception as e:
            print(f"Dataset not yet available: {type(e).__name__}")
    else:
        print(f"Data directory not found")
    
    print("\n--- All tests completed ---\n")