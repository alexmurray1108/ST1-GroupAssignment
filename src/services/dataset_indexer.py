"""
*******************************
Assessment: 3
Author: u3281627
Group Members: u3294093, u3271260
Date: 28/04/2026 (27/04/2026)
Group Assignment
*******************************
"""

from pathlib import Path
import cv2
import pandas as pd

from src.config import RAW_DATA_DIR, SUPPORTED_EXTENSIONS

class DatasetIndexer:
    """Scan selected folder and build DataFrame of image metadata"""
    def __init__(self, data_dir: Path = RAW_DATA_DIR) ->None:
        self.data_dir = data_dir
    
    def build_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with file path, label, width, height, channels"""
        records = []

        for file_path in self.data_dir.rglob("*"):
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            image = cv2.imread(str(file_path))
            if image is None:
                continue

            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) == 3 else 1
            label = file_path.parent.name

            records.append({
                "file_path": str(file_path),
                "label": label,
                "width": width,
                "height": height,
                "channels": channels,
            })
        return pd.DataFrame(records)
    
        
