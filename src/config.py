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

#Image/ file details
IMAGE_SIZE = (32, 32)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}