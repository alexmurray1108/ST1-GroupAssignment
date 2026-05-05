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
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDAService:
    """Generate and save EDA outputs"""

    def __init__(self, dataframe: pd.DataFrame, output_dir: Path) -> None:
        self.dataframe = dataframe
        self.output_dir = output_dir

    def save_class_distribution(self) -> None:
        """Save bar chart of number of images per class"""
        plt.figure(figsize=(12, 6))
        order = self.dataframe["label"].value_counts().index
        sns.countplot(data=self.dataframe, x="label", order=order)
        plt.xticks(rotation=90)
        plt.title("Macroinvertebrate Images per Class")
        plt.tight_layout()
        plt.savefig(self.output_dir / "class_distribution.png")
        plt.close()

    def save_image_size_distribution(self) -> None:
        """Save histograms of image widths and heights"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(self.dataframe["width"], bins=20, ax=axes[0])
        sns.histplot(self.dataframe["height"], bins=20, ax=axes[1])
        axes[0].set_title("Image Width Distribution")
        axes[1].set_title("Image Height Distribution")
        plt.tight_layout()
        plt.savefig(self.output_dir / "image_size_distribution.png")
        plt.close()

    def build_summary(self) -> dict[str, float]:
        """Return key dataset summary statistics"""
        return{
            "total_images": int(len(self.dataframe)),
            "total_classes": int(self.dataframe["label"].nunique()),
            "mean_width": float(self.dataframe["width"].mean()),
            "mean_height": float(self.dataframe["height"].mean())
        }


def save_sample_grid(
    dataframe: pd.DataFrame,
    output_path: Path,
    sample_count: int = 9,
) -> None:
    """Save a 3x3 grid of sample images"""
    sample_df = dataframe.sample(min(sample_count, len(dataframe)), random_state=42)

    _, axes = plt.subplots(3, 3, figsize=(10, 10))

    for ax, (_, row) in zip(axes.flat, sample_df.iterrows()):
        image = cv2.imread(row["file_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.set_title(row["label"])
        ax.axis("off")

    for ax in list(axes.flat)[len(sample_df):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()