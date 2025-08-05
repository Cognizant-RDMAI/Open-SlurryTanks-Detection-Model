Installation Guide
This guide walks you through setting up the **Open Slurry Tank Detection Model** project in your local or cloud environment.
 
> **Note:** This repository is shared for transparency and reproducibility.  
> Code contributions are not accepted at this time.
 
--- 
## Installation Instructions
### 1. Clone the repository
 
First, clone this project to your local machine or cloud environment
```https://github.com/Cognizant-RDMAI/Open-SlurryTanks-Detection-Model```
This will download all project files including the model scripts
 
### 2. Create a Python Virtual Environment (Optional but Recommended)
 
Using a virtual environment avoids conflicts with other Python packages on your system
```
python3 -m venv venv
source venv/bin/activate    #macOS/Linux
venv\Scripts\Activate       #Windows
```
This creates an isolated environment where all dependencies will be installed.

### 3. Install Dependencies
 
All required Python packages are listed in the ```requirements.txt``` file. To install them:
```
pip install -r requirements.txt
```
This includes:
- Ultralytics: YOLOv8 implementation
- Rasterio: For working with GeoTIFF satellite images
- OpenCV: Image Processing
- NumPy: Array and math operations
- Albumentations: For image augmentations

### 3. Prepare Images and Run Detection pipeline
Input satellite images should be placed in:
```
datasets/
└── satellite/
    └── <your_folder>/
        ├── input_image.tif
        └── ...
```
Processed files will be auto auto-created in:
```datasets/satellite/<your_folder>/processing/```
Use the ```detection.ipynb``` notebook to run the full pipeline
