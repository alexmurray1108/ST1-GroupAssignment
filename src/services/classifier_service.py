"""
***********************************
Assessment: 3
Author: u3294093
Group Members: u3281627, u3271260
Date: 28/04/2026
Group Assignment

NOTE: Code is adapted from the Assignment 3 Full Guidance,
with some modifications to better fit the needs of this project.
***********************************
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


class ClassifierService:
    """Train, evaluate, and persist the baseline classification model."""

    def __init__(self, preprocessor, model_output_dir: Path, report_output_dir: Path) -> None:
        print(f"[INIT] Initializing ClassifierService")
        print(f"[INIT] Model output directory: {model_output_dir}")
        print(f"[INIT] Report output directory: {report_output_dir}")
        self.preprocessor = preprocessor
        self.model_output_dir = Path(model_output_dir)
        self.report_output_dir = Path(report_output_dir)
        
        print(f"[INIT] Creating RandomForestClassifier with 200 estimators")
        self.model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
        print(f"[INIT] ClassifierService initialized successfully\n\n")

    def prepare_features(self, dataframe: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Convert indexed file paths into model features and labels."""
        print(f"[PREPARE_FEATURES] Starting feature preparation for {len(dataframe)} samples")
        features = []
        labels = []

        for idx, row in dataframe.iterrows():
            if (idx + 1) % max(1, len(dataframe) // 10) == 0:
                print(f"[PREPARE_FEATURES] Processed {idx + 1}/{len(dataframe)} samples")
            features.append(self.preprocessor.transform(row["file_path"]))
            labels.append(row["label"])

        features_array = np.array(features)
        labels_array = np.array(labels)
        print(f"[PREPARE_FEATURES] Feature preparation completed")
        print(f"[PREPARE_FEATURES] Features shape: {features_array.shape}, Labels shape: {labels_array.shape}\n\n")
        
        return features_array, labels_array

    def train(self, dataframe: pd.DataFrame) -> dict[str, object]:
        """Fit the model and return evaluation outputs."""
        print(f"[TRAIN] Starting model training workflow")
        
        print(f"[TRAIN] Preparing features...")
        X, y = self.prepare_features(dataframe)
        
        print(f"[TRAIN] Splitting data with test_size=0.2, stratified")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        print(f"[TRAIN] Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

        print(f"[TRAIN] Fitting RandomForestClassifier with 200 trees...")
        self.model.fit(X_train, y_train)
        print(f"[TRAIN] Model fitting completed successfully")
        
        print(f"[TRAIN] Generating predictions on test set...")
        predictions = self.model.predict(X_test)
        print(f"[TRAIN] Predictions generated")

        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions, labels=self.model.classes_)
        
        print(f"[TRAIN] Model accuracy: {accuracy:.4f}")
        print(f"[TRAIN] Classification report:\n{report}")
        
        results = {
            "accuracy": accuracy,
            "report": report,
            "confusion_matrix": conf_matrix,
        }

        self.save_report(results, self.report_output_dir)
        self.save_confusion_matrix_plot(results, list(self.model.classes_), self.report_output_dir)
        
        print(f"[TRAIN] Training workflow completed successfully\n\n")
        return results

    def save_model(self, file_name: str = "macro_classifier.joblib") -> Path:
        """Save the trained model to disk."""
        print(f"[SAVE] Creating output directory if needed...")
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = self.model_output_dir / file_name
        print(f"[SAVE] Saving model to {output_path}")
        joblib.dump(self.model, output_path)
        print(f"[SAVE] Model saved successfully to {output_path}\n\n")
        
        return output_path
    
    def save_report(self, results: dict[str, object], output_dir: Path) -> Path:
        """Write the classification report to a text file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "classification_report.txt"
        report_path.write_text(results["report"], encoding="utf-8")
        return report_path
    
    def save_confusion_matrix_plot(
            self,
            results: dict[str, object],
            labels: list[str],
            output_dir: Path,
            ) -> Path:
        """Save the confusion matrix heatmap image."""
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            results["confusion_matrix"],
            annot=False,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels
            )
        plt.title("Confusion Matrix")
        plt.xlabel("predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plot_path = output_dir / "confusion_matrix.png"
        plt.savefig(plot_path)
        plt.close()
        return plot_path
        
        



"""
The Below code is for testing but also allows this module to run on its own separately from main.
This means that if something else doesn't work, I can show that this part still works as intended.
"""
if __name__ == "__main__":
    print("\n--- Classifier Service Test ---\n")
    
    # Create a simple mock preprocessor for testing
    class MockPreprocessor:
        def transform(self, file_path):
            # Return random features for testing
            return np.random.randn(100)
    
    # Initialize the service
    output_dir = Path("outputs/models")
    report_dir = Path("outputs/reports")
    preprocessor = MockPreprocessor()
    service = ClassifierService(preprocessor, output_dir, report_dir)
    
    # Create mock dataframe for testing
    print("\nCreating mock training data...")
    mock_data = {
        "file_path": [f"data/raw/Species_{i}.jpg" for i in range(50)],
        "label": [f"species_{i % 5}" for i in range(50)]
    }
    df = pd.DataFrame(mock_data)
    print(f"Mock dataset created with {len(df)} samples")
    
    # Train the model
    print("\nTraining model...")
    try:
        results = service.train(df)
        print(f"\nResults:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"\nClassification Report:\n{results['report']}")
        
        # Save the trained model
        print("\nSaving model...")
        model_path = service.save_model()
        print(f"Model saved to: {model_path}")
    except Exception as e:
        print(f"Error during training/saving: {type(e).__name__}: {e}")
    
    print("\n--- All tests completed ---\n")