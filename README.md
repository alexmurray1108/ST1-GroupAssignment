# Macro-Invertebrate Classifier

A transfer-learning based image classification system for identifying aquatic macro-invertebrate species using deep neural networks.

## Project Goal

Develop an automated classification system capable of accurately identifying 16 different aquatic macro-invertebrate species from image data using transfer learning with pre-trained deep learning models.

## Features

- **Transfer Learning Pipeline**: Leverages MobileNetV2 pre-trained on ImageNet for efficient classification
- **Multi-class Classification**: Supports 16 aquatic macro-invertebrate species
- **Flexible Data Handling**: Automatic dataset loading and train/validation splitting
- **TensorFlow/Keras Integration**: Modern deep learning framework with straightforward API
- **Modular Service Architecture**: Clean separation between model logic and utilities

## Packages Used

- **TensorFlow** (2.x) - Deep learning framework
- **Keras** - High-level neural networks API
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation and analysis
- **Pillow** - Image processing
- **Matplotlib** - Data visualization
- **Scikit-learn** - Machine learning utilities

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

### GUI Application (To Be Implemented)

```bash
python -m src.gui.app
```

## Folder Structure

```
macro_project/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
│
├── data/                     # Dataset directory
│   ├── raw/                  # Original species images
│   │   ├── Asellus sp/
│   │   ├── Baetidae sp/
│   │   ├── Elmis sp/
│   │   ├── Ephemerellidae/
│   │   ├── Erpobdella sp/
│   │   ├── Gammarus sp/
│   │   ├── Hydropsychidae sp/
│   │   ├── Leptophlebiidae sp/
│   │   ├── Leuctra sp/
│   │   ├── Limnius sp/
│   │   ├── Lymnea sp/
│   │   ├── Nemoura sp/
│   │   ├── Oligochaeta sp/
│   │   ├── Sericostomatidae sp/
│   │   ├── Sialis sp/
│   │   ├── Simuliidae sp/
│   │   └── Sphaerium sp/
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

## Model Architecture

The classifier uses **MobileNetV2** (pre-trained on ImageNet) with:
- **Input**: 224×224 RGB images
- **Base Model**: MobileNetV2 with frozen weights
- **Custom Layers**:
  - Global Average Pooling
  - Dense(128, ReLU)
  - Dense(16, Softmax) for 16-class output
- **Batch Size**: 32
- **Optimizer**: Adam
- **Loss**: Sparse Categorical Crossentropy

## Group Members

- u3294093 (Author)
- u3281627
- u3271260

## Assessment Information

- **Assessment**: 3
- **Date**: 28/04/2024

## Notes

- Python 3.12 or older is required for TensorFlow compatibility
- Ensure all images are in subdirectories matching species names
- Training datasets are automatically split 80/20 for training/validation
- Model training output includes accuracy metrics and loss curves

## License

Group Assignment - Software Technology 1
