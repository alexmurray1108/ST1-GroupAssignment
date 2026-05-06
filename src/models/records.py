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

from dataclasses import dataclass
from pathlib import Path

@dataclass
class ImageRecord:
    """
    Store core metadata for one indexed image
    """

    file_path: Path
    label: str
    width: int
    height: int
    channels: int
