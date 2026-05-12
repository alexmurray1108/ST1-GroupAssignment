"""
*******************************
Assessment: 3
Author: u3281627
Group Members: u3294093, u3271260
Date: 28/04/2026 (30/04/2026)
Group Assignment

NOTE: Code is adapted from the Assignment 3 Full Guidance, 
with some modifications to better fit the needs of this project.
*******************************
"""

from pathlib import Path

#Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

#Data dictionaries
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

#output dictionaries
OUTPUTS_DIR = BASE_DIR / "outputs"
EDA_OUTPUT_DIR = OUTPUTS_DIR / "eda"
MODEL_OUTPUT_DIR = OUTPUTS_DIR / "models"
REPORT_OUTPUT_DIR = OUTPUTS_DIR / "reports"

#Image/ file details
IMAGE_SIZE = (32, 32)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# EDA/reporting settings
PIXEL_ANALYSIS_SAMPLE_SIZE = 10
SAMPLE_GRID_MAX_IMAGES = 16
VERY_SMALL_IMAGE_THRESHOLD = 64
UNUSUAL_ASPECT_RATIO_LOW = 0.5
UNUSUAL_ASPECT_RATIO_HIGH = 2.0

QUALITY_ISSUES_PATH = REPORT_OUTPUT_DIR / "quality_issues.csv"
CLASS_IMBALANCE_REPORT_PATH = REPORT_OUTPUT_DIR / "class_imbalance_report.csv"
