# Macroinvertebrate Image Analysis System

This project analyses macroinvertebrate image data, generates exploratory data analysis outputs, preprocesses images into numeric features, trains a baseline classifier, and launches a console menu for running the workflow from one place.

## Project Goal

The goal is to build an automated system for identifying aquatic macroinvertebrate species from image data. The current implementation indexes the dataset, produces EDA summaries and plots, converts images into feature vectors, and trains a scikit-learn classifier for baseline species prediction.

## Main Features

- Dataset indexing from `data/raw/`
- Class distribution analysis
- Image width and height analysis
- Sample image grid generation
- Image preprocessing into flattened feature vectors
- Baseline classifier training with evaluation metrics
- Console menu interface for running the workflow

## Packages Used

The codebase uses the following Python packages:

- `numpy` 
- `pandas` 
- `opencv-python` (`cv2`)
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `joblib` 

## Installation Instructions

### Prerequisites

- Python 3.14 or Newer is recomended
- `pip`

### Setup

1. Clone the repository and open the project folder:
   ```bash
   git clone https://github.com/alexmurray1108/ST1-GroupAssignment.git
   cd macro_project
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place the raw dataset inside `data/raw/`, keeping one folder per species.

## How to Run Stage 1 and Stage 2

The workflow is driven by `src/main.py`, which runs the stages in order.

### Stage 1: Dataset Indexing and EDA

Stage 1 scans `data/raw/`, builds a dataframe of image metadata, prints a dataset summary, and saves EDA outputs into `outputs/eda/`.

Run the full workflow with:
```bash
python -m src.main
```

This executes the Stage 1 methods that:

- build the indexed dataset
- print summary statistics
- save the class distribution plot
- save the image size distribution plot
- save a sample image grid

### Stage 2: Preprocessing and Classifier Training

Stage 2 converts raw images into fixed-size grayscale feature vectors, saves them to `data/processed/features.npy` and `data/processed/labels.npy`, then trains a baseline `RandomForestClassifier` and saves the model to `outputs/models/macro_classifier.joblib`.

The same command launches Stage 2 automatically after Stage 1:
```bash
python -m src.main
```

## How to Launch the Console Menu Application

Run the main module:
```bash
python -m src.main
```

This starts the console menu after the workflow methods finish running. The menu currently provides options for dataset summary, EDA generation, classifier training, image prediction, and exit.

Note: this codebase does not include a separate Tkinter GUI module. The available interface in the current source is the console menu, and the prediction option is listed there even though the underlying workflow method is not implemented in this snapshot.

## Folder Structure

```text
macro_project/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── Asellus sp/
│   │   ├── Baetidae sp/
│   │   ├── Elmis sp/
│   │   └── ... (species folders)
│   └── processed/
│       ├── features.npy
│       └── labels.npy
├── outputs/
│   ├── eda/
│   ├── models/
│   │   └── macro_classifier.joblib
│   └── reports/
└── src/
    ├── config.py
    ├── console_app.py
    ├── main.py
    ├── models/
    │   └── records.py
    ├── services/
    │   ├── Image_processor.py
    │   ├── classifier_service.py
    │   ├── dataset_indexer.py
    │   ├── eda_service.py
    │   └── workflow_service.py
    └── utils/
        └── plotting.py
```

## Group Members

- u3294093
- u3281627
- u3271260

## Notes

- Raw images should be stored as JPEG or PNG files inside the species folders under `data/raw/`
- The processed feature files in `data/processed/` are created automatically when Stage 2 runs
- If you add more dependencies later, update the installation command or create a `requirements.txt` file for the project
