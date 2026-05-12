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

from config import RAW_DATA_DIR, SUPPORTED_EXTENSIONS

class DatasetIndexer:
    """Scan selected folder and build DataFrame of image metadata"""
    def __init__(self, data_dir: Path = RAW_DATA_DIR) ->None:
        self.data_dir = data_dir
    
    def build_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with file path, label, width, height, channels, and quality metadata."""
        records = []

        for file_path in self.data_dir.rglob("*"):
            file_extension = file_path.suffix.lower()
            if file_extension not in SUPPORTED_EXTENSIONS:
                continue

            image = cv2.imread(str(file_path))
            readable = image is not None
            width = None
            height = None
            channels = None
            aspect_ratio = None

            if readable:
                height, width = image.shape[:2]
                channels = image.shape[2] if len(image.shape) == 3 else 1
                aspect_ratio = width / height if height else None

            label = file_path.parent.name

            records.append({
                "file_path": str(file_path),
                "label": label,
                "width": width,
                "height": height,
                "channels": channels,
                "file_extension": file_extension,
                "readable": readable,
                "aspect_ratio": aspect_ratio,
            })
        return pd.DataFrame(records)
    
        
