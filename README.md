# Macro-Invertebrate Classifier

A transfer-learning based image classification system for identifying aquatic macro-invertebrate species using deep neural networks.

## Project Goal

Develop an automated classification system capable of accurately identifying 16 different aquatic macro-invertebrate species from image data using transfer learning with pre-trained deep learning models.

## Installation

### Prerequisites
- Python 3.12 or older (required for TensorFlow compatibility)
- pip (Python package manager)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/alexmurray1108/ST1-GroupAssignment.git
   cd macro_project
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Workflow

### Stage 1: Data Preparation & Exploration

1. Place raw images in `data/raw/` organized by species folder:
   ```
   data/raw/
   ├── Asellus sp/
   ├── Baetidae sp/
   ├── Elmis sp/
   └── ... (16 species total)
   ```

2. Run exploratory data analysis:
   ```bash
   python -m src.services.classifier_service
   ```

3. Processed datasets will be saved to `data/processed/`

### Stage 2: Model Training & Evaluation

1. **Build and train the model:**
   ```bash
   python -m src.services.classifier_service
   ```

2. **Generate training reports:**
   - Training logs saved to `outputs/reports/`
   - Model checkpoints saved to `outputs/models/`
   - EDA visualizations saved to `outputs/eda/`

3. **Evaluate performance:**
   - View accuracy metrics and loss curves in generated reports
   - Check confusion matrices in `outputs/reports/`

## Launching the Application

### Console Menu (To Be Implemented)

```bash
python -m src.main
```

## Folder Structure

```
macro_project/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
│
├── data/                     # Dataset directory
│   ├── raw/                  # Original species images
│   └── processed/            # Preprocessed datasets
│
├── outputs/                  # Generated outputs
│   ├── eda/                  # Exploratory data analysis reports
│   ├── models/               # Trained model checkpoints
│   └── reports/              # Training metrics and summaries
│
└── src/                      # Source code
    ├── services/
    │   ├── classifier_service.py  # Transfer learning model service
    │   └── classifier_service_explanation.txt
    ├── models/               # Custom model architectures (future)
    └── utils/                # Utility functions (future)
```

## Group Members
- u3294093
- u3281627
- u3271260

## Notes

- Python 3.12 or older is required for TensorFlow compatibility
- 
