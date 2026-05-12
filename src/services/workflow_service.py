"""
*******************************
Assessment: 3
Author: u3281627
Group Members: u3294093, u3271260
Date: 28/04/2026 (27/04/2026)
Group Assignment

NOTE: Code is adapted from the Assignment 3 Full Guidance, 
with some modifications to better fit the needs of this project.
*******************************
"""

from pathlib import Path
import pandas as pd
import joblib

from config import EDA_OUTPUT_DIR, MODEL_OUTPUT_DIR, RAW_DATA_DIR, IMAGE_SIZE, REPORT_OUTPUT_DIR, SUPPORTED_EXTENSIONS
from services.dataset_indexer import DatasetIndexer
from services.classifier_service import ClassifierService
from services.eda_service import EDAService, save_sample_grid
from services.Image_processor import ImagePreprocessor
import numpy as np

class WorkflowService:
    """
    Workflow for Stage 1: indexing, summary, and EDA generation
    """

    def __init__(self) -> None:
        EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.indexer = DatasetIndexer()
        self.dataframe:pd.DataFrame | None = None

    def load_dataframe(self) -> pd.DataFrame:
        """Load and save the indexed dataset"""
        if self.dataframe is None:
            self.dataframe = self.indexer.build_dataframe()
        return self.dataframe
    
    def show_summary(self) -> dict[str, float]:
        """Print summary"""
        df = self.load_dataframe()
        eda = EDAService(df, EDA_OUTPUT_DIR)
        summary = eda.build_summary()
        print(summary)
        return summary
    
    def generate_eda(self) -> None:
        """Generate and save outputs"""
        df = self.load_dataframe()
        eda = EDAService(df, EDA_OUTPUT_DIR)
        eda.save_class_distribution()
        eda.save_image_size_distribution()
        save_sample_grid(df, EDA_OUTPUT_DIR / "sample_grid.png")
        print("EDA outputs generated successfully.")

    def preprocess_images(self) -> tuple[str, str] | None:
        """
        Preprocess raw images and save features/labels to data/processed.
        Returns a tuple of (features_path, labels_path) or None if no images.
        """
        preprocessor = ImagePreprocessor(image_size=IMAGE_SIZE)
        raw_files = [p for p in RAW_DATA_DIR.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]

        if not raw_files:
            print("[PREPROCESS] No raw images found to preprocess")
            return None

        features = preprocessor.batch_transform(raw_files)
        labels = [p.parent.name for p in raw_files]
        processed_dir = RAW_DATA_DIR.parent / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        features_path = processed_dir / "features.npy"
        labels_path = processed_dir / "labels.npy"
        np.save(features_path, features)
        np.save(labels_path, np.array(labels))
        print(f"[PREPROCESS] Saved {len(raw_files)} feature vectors to {processed_dir}\n\n")
        return str(features_path), str(labels_path)

    def train_classifier(self) -> str:
        """
        Train and save the classifier model using the indexed dataset.
        """
        print("[TRAIN_CLASSIFIER] Starting classifier training workflow")
        
        print("[TRAIN_CLASSIFIER] Preprocessing images and saving to data/processed/...")
        self.preprocess_images()
        
        df = self.load_dataframe()
        print(f"[TRAIN_CLASSIFIER] Loaded dataframe with {len(df)} samples")
        
        preprocessor = ImagePreprocessor(image_size=IMAGE_SIZE)
        service = ClassifierService(preprocessor, MODEL_OUTPUT_DIR, REPORT_OUTPUT_DIR)
        
        print(f"[TRAIN_CLASSIFIER] Training classifier...")
        results = service.train(df)
        print(f"[TRAIN_CLASSIFIER] Model accuracy: {results['accuracy']:.4f}")
        
        model_path = service.save_model()
        print(f"[TRAIN_CLASSIFIER] Classifier training completed\n\n")
        return str(model_path)

    def predict_image(self, image_path: str) -> str:
        """
        Predict the macroinvertebrate class for a single image path.
        """
        model_path = MODEL_OUTPUT_DIR / "macro_classifier.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {model_path}. Please run option 3 first."
            )

        preprocessor = ImagePreprocessor(image_size=IMAGE_SIZE)
        model = joblib.load(model_path)

        features = preprocessor.transform(image_path)
        probabilities = model.predict_proba([features])[0]
        predicted_index = int(np.argmax(probabilities))
        prediction = model.classes_[predicted_index]
        confidence = probabilities[predicted_index]

        print(f"Predicted class: {prediction}")
        print(f"Prediction confidence: {confidence:.2%}")
        return str(prediction)