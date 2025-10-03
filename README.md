
Dependencies
To run the code, ensure you have the following Python libraries installed:

Python: Version 3.6 or higher
PyTorch: For building and training the MedT model (torch, torchvision)
NumPy: For numerical operations and array handling (numpy)
OpenCV: For image processing and augmentation (opencv-python, opencv-contrib-python)
scikit-learn: For metrics calculation and data splitting (scikit-learn)
Matplotlib: For plotting training history and visualizations (matplotlib)
TIFFfile: For reading TIFF image files (tifffile)
gc: For garbage collection to manage memory
os: For file system operations
random: For random number generation and data augmentation

You can install the required dependencies using pip with the following command:
pip install torch torchvision numpy opencv-python opencv-contrib-python scikit-learn matplotlib tifffile

Hardware Requirements

CPU/GPU: The code supports both CPU and GPU training. For faster training, a CUDA-compatible GPU is recommended.
Memory: At least 8GB RAM for CPU training; 16GB or more recommended for GPU training.
Storage: Ensure sufficient disk space for input images, labels, and model outputs (e.g., checkpoints, visualizations).

Setup Instructions

Prepare Data:

Place your input images (TIFF format) in the directory specified by SLICE_DIR 
Place corresponding segmentation masks (TIFF format) in the directory specified by LABEL_DIR
Ensure the number of image and mask files match and are correctly paired.


Configure Output Directory:

The model outputs (checkpoints, training history, and visualizations) will be saved to OUTPUT_DIR


Install Dependencies:

Run the pip command above to install all required libraries.
Verify PyTorch installation with CUDA support if using a GPU:python -c "import torch; print(torch.cuda.is_available())"



Configuration Parameters
Key parameters can be adjusted as needed:
﻿
PATCH_SIZE: Size of image patches
PATCHES_PER_IMAGE: Number of patches extracted per image 
BATCH_SIZE: Number of samples per batch 
EPOCHS: Maximum number of training epochs 
NUM_CLASSES: Number of segmentation classes 
LEARNING_RATE: Initial learning rate 
VALIDATION_SPLIT: Fraction of data for validation 
TEST_SPLIT: Fraction of data for testing 
ROTATION_RANGE, ZOOM_RANGE, SHEAR_RANGE, HORIZONTAL_FLIP, VERTICAL_FLIP: Data augmentation parameters.
