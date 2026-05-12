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

import os
import tempfile
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "macro_stage1_stage3_matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    CLASS_IMBALANCE_REPORT_PATH,
    EDA_OUTPUT_DIR,
    PIXEL_ANALYSIS_SAMPLE_SIZE,
    QUALITY_ISSUES_PATH,
    REPORT_OUTPUT_DIR,
    SAMPLE_GRID_MAX_IMAGES,
    UNUSUAL_ASPECT_RATIO_HIGH,
    UNUSUAL_ASPECT_RATIO_LOW,
    VERY_SMALL_IMAGE_THRESHOLD,
)


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_int(value: object) -> int | None:
    if pd.isna(value):
        return None
    return int(round(float(value)))


def _safe_round(value: object, digits: int = 2) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), digits)


def _coerce_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    frame = dataframe.copy()

    if "file_extension" not in frame.columns and "file_path" in frame.columns:
        frame["file_extension"] = frame["file_path"].astype(str).map(lambda value: Path(value).suffix.lower())

    if "readable" not in frame.columns:
        frame["readable"] = True

    if "aspect_ratio" not in frame.columns:
        width = pd.to_numeric(frame.get("width"), errors="coerce") if "width" in frame.columns else pd.Series(dtype="float64")
        height = pd.to_numeric(frame.get("height"), errors="coerce") if "height" in frame.columns else pd.Series(dtype="float64")
        frame["aspect_ratio"] = width / height

    for column in ("width", "height", "channels", "aspect_ratio"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["readable"] = frame["readable"].fillna(False).astype(bool)
    return frame


def _write_sample_grid(dataframe: pd.DataFrame, output_path: Path, sample_count: int) -> Path:
    readable = dataframe[dataframe["readable"].astype(bool)]
    if readable.empty:
        raise ValueError("No readable images were available for sample grid generation.")

    samples = readable.groupby("label", group_keys=False).head(1)
    if len(samples) > sample_count:
        samples = samples.sample(sample_count, random_state=42)

    samples = samples.reset_index(drop=True)
    columns = min(4, len(samples))
    rows = int(np.ceil(len(samples) / columns))

    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.4 * rows))
    axes_array = np.array(axes).reshape(-1)

    for axis in axes_array:
        axis.axis("off")

    for axis, (_, row) in zip(axes_array, samples.iterrows()):
        image = cv2.imread(str(row["file_path"]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axis.imshow(rgb_image)
        axis.set_title(str(row["label"]), fontsize=10)
        axis.axis("off")

    fig.suptitle("Representative Sample Images by Class")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


class EDAService:
    """Generates Stage 1 EDA tables, charts, quality checks, and reports."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        eda_output_dir: Path = EDA_OUTPUT_DIR,
        report_output_dir: Path = REPORT_OUTPUT_DIR,
    ) -> None:
        """Prepare the EDA service with a dataset index."""
        self.dataframe = _coerce_dataframe(dataframe)
        self.eda_output_dir = Path(eda_output_dir)
        self.report_output_dir = Path(report_output_dir)
        readable_mask = self.dataframe["readable"].astype(bool)
        self.readable_dataframe = self.dataframe[readable_mask]

        _ensure_directory(self.eda_output_dir)
        _ensure_directory(self.report_output_dir)
        sns.set_theme(style="whitegrid")

    def generate_all_outputs(self) -> list[Path]:
        """Generate all required EDA outputs and return their paths."""
        if self.dataframe.empty:
            raise ValueError("The dataset index is empty.")

        return [
            self.generate_dataset_summary(),
            self.generate_class_distribution_chart(),
            self.generate_image_size_distribution_chart(),
            self.generate_width_height_scatter_plot(),
            self.generate_sample_image_grid(),
            self.generate_width_by_class_boxplot(),
            self.generate_height_by_class_boxplot(),
            self.generate_pixel_intensity_histogram(),
            self.generate_image_quality_issues(),
            self.generate_class_imbalance_report(),
        ]

    def generate_dataset_summary(self) -> Path:
        """Save a high-level dataset summary CSV."""
        class_counts = self.dataframe["label"].value_counts().sort_index()
        readable = self.readable_dataframe
        supported_types = ", ".join(sorted(map(str, self.dataframe["file_extension"].dropna().unique())))

        summary_rows = [
            ("total_images", len(self.dataframe)),
            ("total_classes", self.dataframe["label"].nunique()),
            ("images_per_class", self._format_class_counts(class_counts)),
            ("mean_width", _safe_round(readable["width"].mean())),
            ("mean_height", _safe_round(readable["height"].mean())),
            ("min_width", _safe_int(readable["width"].min())),
            ("max_width", _safe_int(readable["width"].max())),
            ("min_height", _safe_int(readable["height"].min())),
            ("max_height", _safe_int(readable["height"].max())),
            ("number_of_unreadable_files", int((~self.dataframe["readable"]).sum())),
            ("supported_file_types_found", supported_types),
        ]

        summary = pd.DataFrame(summary_rows, columns=["metric", "value"])
        output_path = self.eda_output_dir / "dataset_summary.csv"
        summary.to_csv(output_path, index=False)
        return output_path

    def generate_class_distribution_chart(self) -> Path:
        """Save a bar chart showing the number of images in each class."""
        class_counts = self.dataframe["label"].value_counts().sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=class_counts.index, y=class_counts.values, color="#4C78A8")
        plt.title("Class Balance: Images per Macroinvertebrate Class")
        plt.xlabel("Class label")
        plt.ylabel("Number of images")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()

        output_path = self.eda_output_dir / "class_distribution.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path

    def generate_image_size_distribution_chart(self) -> Path:
        """Save width and height histograms in one figure."""
        readable = self._require_readable_images()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(readable["width"], bins=20, ax=axes[0], color="#4C78A8")
        axes[0].set_title("Image Width Distribution")
        axes[0].set_xlabel("Width in pixels")

        sns.histplot(readable["height"], bins=20, ax=axes[1], color="#F58518")
        axes[1].set_title("Image Height Distribution")
        axes[1].set_xlabel("Height in pixels")

        fig.tight_layout()
        output_path = self.eda_output_dir / "image_size_distribution.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    def generate_width_height_scatter_plot(self) -> Path:
        """Save a scatter plot of image width versus height."""
        readable = self._require_readable_images()

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=readable, x="width", y="height", hue="label", alpha=0.75)
        plt.title("Image Width Versus Height")
        plt.xlabel("Width in pixels")
        plt.ylabel("Height in pixels")
        plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        output_path = self.eda_output_dir / "width_height_scatter.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path

    def generate_sample_image_grid(self) -> Path:
        """Save a grid of representative readable sample images."""
        output_path = self.eda_output_dir / "sample_image_grid.png"
        return _write_sample_grid(self.dataframe, output_path, SAMPLE_GRID_MAX_IMAGES)

    def generate_width_by_class_boxplot(self) -> Path:
        """Save a boxplot comparing image widths by class."""
        readable = self._require_readable_images()

        plt.figure(figsize=(11, 6))
        sns.boxplot(data=readable, x="label", y="width", color="#72B7B2")
        plt.title("Image Width by Class")
        plt.xlabel("Class label")
        plt.ylabel("Width in pixels")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()

        output_path = self.eda_output_dir / "width_by_class_boxplot.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path

    def generate_height_by_class_boxplot(self) -> Path:
        """Save a boxplot comparing image heights by class."""
        readable = self._require_readable_images()

        plt.figure(figsize=(11, 6))
        sns.boxplot(data=readable, x="label", y="height", color="#54A24B")
        plt.title("Image Height by Class")
        plt.xlabel("Class label")
        plt.ylabel("Height in pixels")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()

        output_path = self.eda_output_dir / "height_by_class_boxplot.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path

    def generate_pixel_intensity_histogram(self) -> Path:
        """Save a grayscale pixel intensity histogram from sampled images."""
        readable = self._require_readable_images()
        sample = readable.head(PIXEL_ANALYSIS_SAMPLE_SIZE)
        intensity_values: list[int] = []

        for _, row in sample.iterrows():
            grayscale = cv2.imread(str(row["file_path"]), cv2.IMREAD_GRAYSCALE)
            if grayscale is not None:
                intensity_values.extend(grayscale.flatten().tolist())

        if not intensity_values:
            raise ValueError("No readable pixels were available for intensity analysis.")

        plt.figure(figsize=(9, 6))
        sns.histplot(intensity_values, bins=50, color="#B279A2")
        plt.title("Sampled Grayscale Pixel Intensity Distribution")
        plt.xlabel("Pixel intensity, 0 dark to 255 bright")
        plt.ylabel("Frequency")
        plt.tight_layout()

        output_path = self.eda_output_dir / "pixel_intensity_histogram.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path

    def generate_image_quality_issues(self) -> Path:
        """Save image quality flags to CSV."""
        records: list[dict[str, object]] = []
        for _, row in self.dataframe.iterrows():
            issues = []
            if not bool(row["readable"]):
                issues.append("unreadable_or_corrupted")
            if bool(row["readable"]) and (
                self._is_small_image(row["width"]) or self._is_small_image(row["height"])
            ):
                issues.append("very_small_image")
            if bool(row["readable"]) and self._has_unusual_aspect_ratio(row["aspect_ratio"]):
                issues.append("unusual_aspect_ratio")

            if issues:
                records.append(
                    {
                        "file_path": row["file_path"],
                        "label": row["label"],
                        "width": row["width"],
                        "height": row["height"],
                        "aspect_ratio": row["aspect_ratio"],
                        "issues": "; ".join(issues),
                    }
                )

        issues_frame = pd.DataFrame(records)
        output_path = QUALITY_ISSUES_PATH
        _ensure_directory(output_path.parent)
        issues_frame.to_csv(output_path, index=False)
        return output_path

    def generate_class_imbalance_report(self) -> Path:
        """Save a class balance report with counts and percentages."""
        class_counts = self.dataframe["label"].value_counts().sort_values(ascending=False)
        total = int(class_counts.sum()) if not class_counts.empty else 0
        max_count = int(class_counts.max()) if not class_counts.empty else 0

        report = pd.DataFrame(
            {
                "label": class_counts.index,
                "count": class_counts.values,
                "percentage": [round((count / total) * 100, 2) if total else 0.0 for count in class_counts.values],
                "imbalance_ratio_vs_largest": [round(max_count / count, 2) if count else None for count in class_counts.values],
            }
        )
        output_path = CLASS_IMBALANCE_REPORT_PATH
        _ensure_directory(output_path.parent)
        report.to_csv(output_path, index=False)
        return output_path

    def save_class_distribution(self) -> Path:
        """Compatibility wrapper for the original workflow API."""
        return self.generate_class_distribution_chart()

    def save_image_size_distribution(self) -> Path:
        """Compatibility wrapper for the original workflow API."""
        return self.generate_image_size_distribution_chart()

    def build_summary(self) -> dict[str, object]:
        """Return key dataset summary statistics."""
        class_counts = self.dataframe["label"].value_counts().sort_index()
        readable = self.readable_dataframe

        return {
            "total_images": int(len(self.dataframe)),
            "total_classes": int(self.dataframe["label"].nunique()),
            "images_per_class": self._format_class_counts(class_counts),
            "mean_width": _safe_round(readable["width"].mean()),
            "mean_height": _safe_round(readable["height"].mean()),
            "min_width": _safe_int(readable["width"].min()),
            "max_width": _safe_int(readable["width"].max()),
            "min_height": _safe_int(readable["height"].min()),
            "max_height": _safe_int(readable["height"].max()),
            "number_of_unreadable_files": int((~self.dataframe["readable"]).sum()),
            "supported_file_types_found": ", ".join(sorted(map(str, self.dataframe["file_extension"].dropna().unique()))),
        }

    def _require_readable_images(self) -> pd.DataFrame:
        if self.readable_dataframe.empty:
            raise ValueError("The dataset does not contain any readable images.")
        return self.readable_dataframe

    def _is_small_image(self, value: object) -> bool:
        if pd.isna(value):
            return False
        return float(value) < VERY_SMALL_IMAGE_THRESHOLD

    def _has_unusual_aspect_ratio(self, value: object) -> bool:
        if pd.isna(value):
            return False
        ratio = float(value)
        return ratio < UNUSUAL_ASPECT_RATIO_LOW or ratio > UNUSUAL_ASPECT_RATIO_HIGH

    def _quality_issue_counts(self) -> dict[str, int]:
        counts = {
            "unreadable_or_corrupted": 0,
            "very_small_image": 0,
            "unusual_aspect_ratio": 0,
        }
        for _, row in self.dataframe.iterrows():
            if not bool(row["readable"]):
                counts["unreadable_or_corrupted"] += 1
            if bool(row["readable"]) and (
                self._is_small_image(row["width"]) or self._is_small_image(row["height"])
            ):
                counts["very_small_image"] += 1
            if bool(row["readable"]) and self._has_unusual_aspect_ratio(row["aspect_ratio"]):
                counts["unusual_aspect_ratio"] += 1
        return counts

    def _format_class_counts(self, class_counts: pd.Series) -> str:
        return "; ".join(f"{label}: {int(count)}" for label, count in class_counts.items())


def save_sample_grid(
    dataframe: pd.DataFrame,
    output_path: Path,
    sample_count: int = SAMPLE_GRID_MAX_IMAGES,
) -> Path:
    """Save a sample image grid while preserving the old helper API."""
    prepared = _coerce_dataframe(dataframe)
    _ensure_directory(Path(output_path).parent)
    return _write_sample_grid(prepared, Path(output_path), sample_count)