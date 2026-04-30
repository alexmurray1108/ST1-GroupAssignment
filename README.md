# Macro Project - Transfer Learning Classifier

A comprehensive transfer learning solution for image classification of macro-invertebrates using deep neural networks.

## Project Overview

This project implements a transfer learning pipeline using **MobileNetV2** for classifying macro-invertebrate species from image data. The solution demonstrates best practices in machine learning engineering including modular design, comprehensive error handling, logging, and validation.

### Key Features

- **Transfer Learning**: Leverages pre-trained MobileNetV2 weights from ImageNet
- **Robust Error Handling**: Comprehensive validation and error checking at all levels
- **Logging & Monitoring**: Integrated logging for debugging and tracking execution
- **Type Hints**: Full type annotations for better code clarity and IDE support
- **Comprehensive Documentation**: Detailed docstrings with examples
- **Unit Tests**: Extensive test coverage for all major components

## Directory Structure

```
macro_project/
├── data/
│   ├── raw/                 # Raw image data organized by species
│   │   ├── Asellus sp/
│   │   ├── Baetidae sp/
│   │   ├── ...
│   │   └── Simuliidae sp/
│   └── processed/           # Preprocessed artifacts
├── src/
│   ├── services/
│   │   ├── classifier_service.py    # Main transfer learning service
│   │   └── test_classifier_service.py # Comprehensive unit tests
│   ├── models/              # Model definitions and utilities
│   └── utils/               # Utility functions
├── outputs/
│   ├── eda/                 # Exploratory data analysis outputs
│   ├── models/              # Trained models and weights
│   └── reports/             # Analysis reports
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation (this file)
└── MANUAL_TESTING.md        # Manual testing procedures
```

## Installation

### Prerequisites

- Python 3.8+
- pip or conda package manager

### Setup

1. **Clone/navigate to the project directory**
   ```bash
   cd macro_project
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Core Module: TransferLearningService

### Overview

The `TransferLearningService` class encapsulates the complete workflow for building and training transfer learning models.

### Model Architecture

The service builds a sequential model with the following layers:

1. **Rescaling Layer**: Normalizes pixel values to [0, 1]
2. **MobileNetV2 Base**: Pre-trained feature extractor (ImageNet weights)
3. **Global Average Pooling**: Reduces spatial dimensions
4. **Dense Layer**: 128 units with ReLU activation for feature refinement
5. **Output Layer**: Softmax activation with class_count units

### Usage Examples

#### Basic Model Building

```python
from src.services.classifier_service import TransferLearningService

# Initialize service
service = TransferLearningService(
    image_size=(224, 224),
    batch_size=32
)

# Build model for multi-class classification
model = service.build_model(class_count=16)

# Print model summary
model.summary()
```

#### Dataset Creation

```python
# Load training and validation datasets
train_ds, val_ds = service.build_datasets(
    train_dir='data/raw/',
    validation_split=0.2
)

# Get dataset statistics
print(f"Batch size: {service.batch_size}")
print(f"Image size: {service.image_size}")
```

#### Training a Model

```python
# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    verbose=1
)

# Evaluate on test set
loss, accuracy = model.evaluate(val_ds)
print(f"Validation Accuracy: {accuracy:.4f}")
```

### Class: TransferLearningService

#### Constructor Parameters

- **image_size** (tuple[int, int]): Target image dimensions. Default: (224, 224)
- **batch_size** (int): Batch size for dataset loading. Default: 32

#### Methods

##### `build_datasets(train_dir, validation_split=0.2)`

Creates training and validation datasets from a directory structure.

**Parameters:**
- `train_dir` (Path | str): Root directory containing class subdirectories
- `validation_split` (float): Fraction for validation (0 < x < 1). Default: 0.2

**Returns:**
- tuple[tf.data.Dataset, tf.data.Dataset]: Training and validation datasets

**Raises:**
- `FileNotFoundError`: If training directory doesn't exist
- `NotADirectoryError`: If path is not a directory
- `ValueError`: If validation_split is out of range or directory is empty

##### `build_model(class_count)`

Builds and compiles a transfer learning model.

**Parameters:**
- `class_count` (int): Number of output classes (≥ 2)

**Returns:**
- keras.Model: Compiled model ready for training

**Raises:**
- `ValueError`: If class_count < 2
- `RuntimeError`: If output layer doesn't match class_count

##### `get_model()`

Retrieves the currently built model.

**Returns:**
- Optional[keras.Model]: Built model or None

## Testing

### Running Unit Tests

```bash
cd src/services
python -m pytest test_classifier_service.py -v
```

### Test Coverage

The test suite includes:
- **Initialization tests**: Parameter validation and defaults
- **Dataset building tests**: Error handling and edge cases
- **Model building tests**: Output validation and error conditions
- **Validation method tests**: Input checking

### Smoke Test

A quick sanity check is built into the module:

```bash
python src/services/classifier_service.py
```

This will:
1. Instantiate the service
2. Build a simple 2-class model
3. Verify compilation succeeds

## Error Handling & Validation

The service implements comprehensive error handling:

- **Parameter Validation**: All inputs are validated at initialization
- **Path Validation**: Directory existence and format checks
- **Data Validation**: Empty directory detection
- **Type Checking**: Full type hint support
- **Logging**: Detailed logging at DEBUG and INFO levels

## Logging

The module uses Python's built-in logging. Configure logging in your application:

```python
import logging

# Set to DEBUG for detailed output
logging.basicConfig(level=logging.DEBUG)

# Now use the service normally
service = TransferLearningService()
```

Log output includes:
- Service initialization
- Dataset creation steps
- Model architecture details
- Error conditions and validation failures

## Dependencies

- **tensorflow**: Deep learning framework
- **keras**: Neural network API
- **numpy**: Numerical computing
- **pathlib**: Cross-platform path handling

See `requirements.txt` for complete list.

## Development Notes

### Code Quality

- Full type hints for improved IDE support and code clarity
- PEP 8 compliant formatting
- Comprehensive docstrings (Google style)
- Extensive error messages for debugging

### Best Practices Implemented

- ✅ Input validation and error handling
- ✅ Logging for monitoring and debugging
- ✅ Type hints throughout
- ✅ Comprehensive documentation
- ✅ Unit tests with good coverage
- ✅ Separation of concerns
- ✅ Clear exception messages
- ✅ Resource management

## Future Enhancements

Potential improvements for future development:
- Model checkpointing and early stopping
- Data augmentation pipelines
- Hyperparameter tuning utilities
- Model evaluation and metrics
- Export/import functionality
- Batch prediction utilities

## Author & Attribution

**Assessment**: 3
**Author**: u3294093
**Group Members**: u3281627, u3271260
**Date**: 28/04/2024

## License

Educational project
