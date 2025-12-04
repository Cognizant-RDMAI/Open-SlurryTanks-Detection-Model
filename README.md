# Open Slurry Tank Detection Model


## 1. Introduction
This project introduces a custom-trained object detection model based on YOLOv8, specifically designed to identify and geolocate circular slurry tanks from high-resolution aerial and satellite imagery. At this point, the model is trained on only circular type open slurry tanks. Other slurry storage structures including slurry lagoons and pits have not been considered. Slurry tanks, often used for agricultural waste storage, are scattered features that can be challenging to detect at scale. Our work addresses this by building a complete geospatial pipeline, from data collection and annotation to model training and spatial post-processing.

The model was trained using a curated dataset that includes both manually annotated imagery and publicly available datasets, enabling it to generalize across diverse landscapes and imagery sources. We used various data augmentation techniques to enhance model robustness given the limited number of training samples. YOLOv8 was selected for its high accuracy, real-time performance, and compatibility with large-scale imagery analysis.

Beyond detection, the output of the model includes precise geospatial coordinates for each identified slurry tank, allowing seamless integration into GIS workflows. This capability supports downstream applications such as environmental monitoring, regulatory compliance, infrastructure inventory, and policy planning.

### 1.1 Motivation
Slurry tanks are one of the main types of structure for storing livestock manures. The manure stored within these structures potentially presents a risk to water, soil and air due to subsequent application of the manure to land, emissions to air during storage and the risk of leakage from the tank and handling systems entering waterbodies. 
With the rise of remote sensing and artificial intelligence, detecting slurry tanks using aerial or satellite imagery has not only become feasible but also scalable. Instead of manually inspecting thousands of aerial images, an object detection model can scan imagery for specific features in near real-time, enabling large-scale environmental monitoring.

### 1.2 RDMAI Overview
River Deep Mountain AI is an innovation project funded by the Ofwat Innovation Fund working collaboratively to develop open-source AI/ML models that can inform effective actions to tackle waterbody pollution.

### 1.3 Purpose and Functionality
The primary purpose of this project is to develop an object detection model capable of identifying and geolocating **open-type slurry tanks** from high-resolution satellite and aerial imagery. These tanks are key agricultural infrastructure elements, and mapping their locations could help support a range of downstream applications such as environmental and nutrient pollution monitoring, regulatory compliance and enforcement, agricultural land use assessment and infrastructure inventory and planning
 
The model was designed to work as part of a fully automated geospatial pipeline. It can process large imagery datasets using batch inference, enabling analysis at regional and national scales It performs the following core functions:
 
- **Detection**: Identify slurry tanks in aerial/satellite images using a custom-trained YOLOv8 model.
- **Geolocation**: Convert pixel-based bounding boxes into real-world geographic coordinates using metadata from the source GeoTIFFs.
- **Integration**: Export outputs as GeoJSON files compatible with common GIS platforms such as QGIS, ArcGIS, and Leaflet.

## 2. Installation Instructions
To get started with the slurry tank detection model:
### 2.1 Clone the repository
First, clone this project to your local machine or cloud environment
```https://github.com/Cognizant-RDMAI/Open-SlurryTanks-Detection-Model```
This will download all project files including the model scripts.

### 2.2 Create a Python Virtual Environment (Optional but Highly Recommended)
 
Using a virtual environment avoids conflicts with other Python packages on your system
```
python3 -m venv venv
source venv/bin/activate    #macOS/Linux
venv\Scripts\Activate       #Windows
```
This creates an isolated environment where all dependencies will be installed.

### 2.3 Install Dependencies
 
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

### 2.4 Model Training Pre-requisites

To train the model, we recommend using very high-resolution satellite imagery, ideally with a ground sampling distance of approximately 0.5-0.7m. However, please note that such imagery may be subject to licensing restrictions.
 
During development, we used several high resolution imagery sources, including but not limited to the NAIP tile dataset and licensed Esri basemap imagery strictly for internal purposes.

Use of Esri imagery is subject to specific licensing terms and conditions. We do not recommend using it with third-party tools. Instead, users should strictly adhere to Esri’s licensing agreements. Please refer to the section titled [Additional Resources](#additional-resources) for more details.

### 2.5 Running the model
This sections provides insights on how to run the model.

#### Prepare the input images

Input satellite images should be placed in:
```
datasets/
└── satellite/
    └── <your_folder>/
        ├── input_image.tif
        └── ...
```
Each input image should be a high-resolution `.tif` file with valid georeferencing metadata. We recommend using images with a GSD of 0.3-1 meter for best results. For PNG images, provide the accompanying metadata in form of STAC-style JSON files. Details are given in the ```detection.ipynb``` file.
 Below is an overview of the main files and folders in this project:
```
    Open-SlurryTanks-Detection-Model/
    ├── README.md                 # Project overview and usage instructions
    ├── detection.ipynb           # Inference and post-processing workflow
    ├── slurry_tank_model.ipynb   # Model training, augmentation, and evaluation
    ├── MODEL_CARD.md             # Model details
    ├── INSTALL.md                # Step-by-step installation instructions
    ├── CONTRIBUTING.md           # Contribution guidelines (internal use only)
    ├── CHANGELOG.md              # List of changes and improvements made to the project
    ├── requirements.txt          #python libraries required 
    ├── runs/
    │   └── model_weight/         # Trained YOLOv8 model weights (e.g., best.pt)
    └── LICENSE                   # Licensing information and usage rights
```
The ```slurry_tank_model.ipynb``` contains the code used to develop the model. The ```detections.ipynb``` file contains code to draw inference from the model.

#### Running the Detection Pipeline
 
Run the full satellite image processing and object detection pipeline using `detection.ipynb`. The pipeline performs the following steps:
 
1. **Tiling**: Splits each large GeoTIFF into 512×512 pixel tiles.
2. **Image Enhancement & Conversion**: Applies contrast stretching and converts tiles to PNG format.
3. **YOLOv8 Inference**: Detects slurry tanks in each PNG tile using the trained YOLOv8 model.
4. **Geo-referencing & Export**: Converts bounding boxes to geographic coordinates and exports all detections to a single `.geojson` file.
 
Processed outputs are stored under:```datasets/satellite/<your_folder>/processing/```.

The outputs contain the following:
- PNG tiles with bounding boxes
- YOLO label `.txt` files with confidence scores
- A final `detections.geojson` file ready for use in GIS platforms like QGIS or ArcGIS 
The detection pipeline script also supports multiple `.tif` images in batch mode. 
 
#### Inference Parameters (Customizable in Code)
You can modify the following in the script to suit your dataset:
- `INPUT_FOLDER` – path to folder with `.tif` images
- `TILE_SIZE` – size of image tiles (default is 512)
- `MODEL_PATH` – path to your trained `best.pt` weights
- `CLASS_NAMES` – list of classes your model can detect

### 2.6 Training the Model (Optional) 
If you wish to train or fine-tune the model on your own data, use `slurry_tank_model.ipynb` notebook. It gives a detailed methodology and code on how the model was trained.
 
#### Dataset Structure
Organize your training data into the following format:
```
processed_data
  ├── data.yaml
  ├── test
  ├── train
  └── val
```
Ensure all labels are in **YOLO format** (`.txt`) and aligned with image file names.
 
#### Training Configuration
Open and run the `slurry_tank_model.ipynb` notebook. The training config used in our original setup:
 
- **Model:** YOLOv8l  
- **Epochs:** 300  
- **Batch Size:** 8  
- **Image Size:** 640×640  
- **Optimizer:** AdamW  
- **Learning Rate:** 0.001 → 0.0001  
- **Weight Decay:** 0.0005  
- **Early Stopping:** Enabled (patience=20)  
- **Augmentations:** Mosaic, Flip, HSV, Translate, Dropout, Multiscale  
 
Training was done on GPU with `device=0`.
 
#### Output
Trained weights and logs are saved under: ```runs/detect/train/weights/best.pt```
> For more training details and evaluation metrics, refer to the [Model Card](model_card.md).

> For best results, ensure your input images are high resolution and well-georeferenced. All spatial transformations are handled using the original GeoTIFF’s metadata.This model was trained primarily on high-resolution satellite/aerial imagery corresponding to a **ground sampling distance (GSD) of ~0.5 to 1 meter per pixel**, depending on the region.The model performs best when applied to imagery with **similar spatial resolution and spectral characteristics** to those used during training. Using significantly lower-resolution or multispectral data may degrade detection accuracy.
> We recommend using satellite or aerial imagery that offers:
> - **0.3 to 1 meter per pixel resolution**
> - **RGB bands** (Red, Green, Blue) for visual similarity

## 3. Datasets and Dependencies
This model was trained using a combination of publicly available datasets and a custom-annotated dataset specifically created for slurry tank detection.The below section provides a detailed overview of data collection.

### 3.1 Above Ground Storage Tank Dataset
- Source: [Robinson et al., Scientific Data (2024)](https://www.nature.com/articles/s41597-023-02780-1)
- Description: Over **130,000** annotated storage tanks across the contiguous United States.
- Format: Provided in **Pascal VOC (XML)** format
- The unique classes present in the dataset includes ``` 'closed_roof_tank', 'external_floating_roof_tank', 'narrow_closed_roof_tank', 'sedimentation_tank', 'spherical_tank', 'undefined_object', 'water_tower'``` 
The annotation files in Pascal VOC(XML) format needs to be converted to YOLO format(.txt), to make it compatible with YOLO Object Detection model.
- YOLO (TXT) Format:
Each image get a `.txt` file with the following format per line:```class_id, center_x, center_y, width, height```

- `class_id`: Integer ID of the object class
- `center_x, center_y`: Center of bounding box (normalised to image dimensions)
- `width, height`: Bounding box size (normalised to image dimensions)
  
### 3.2 Custom Slurry Tank Dataset Collection
 
To support the slurry tank detection task, a custom dataset was created using a structured geospatial workflow. The collection process involved the following steps:
 
**1. Identifying Slurry Tank Locations**
 
- Slurry tank locations were sourced using **OpenStreetMap (OSM)** datasets.
- Data was downloaded from: [https://download.geofabrik.de](https://download.geofabrik.de)
- Locations were selected based on proximity to agricultural areas and visible storage structures.
 
***Number of identified locations:***
- **England** – 93  
- **Denmark** – 79  
- **Wales** – 101
 
**2. High-Resolution Imagery**

- Tile size: `256 × 256` pixels  
- Ground resolution: `0.5–0.7 meters per pixel`  
- Ground coverage per tile: `150–200 m²`.This resolution provided sufficient detail to visually distinguish open-type slurry tanks.
 
**3. Annotating Slurry Tanks**
 
- Manual annotation was performed using **Label Studio**: [https://labelstud.io](https://labelstud.io)
- Only **open-type** slurry tanks were labeled.
- Annotations were exported directly in **YOLO format**. 
**Annotation criteria:**
- Circular shape with visible contents from top view.
- No overhead cover or roof.
- Commonly adjacent to farms or livestock barns.
 
This dataset was later merged with a publicly available storage tank dataset to support robust training of the YOLOv8 object detection model.

### 3.3 Dependencies
To ensure smooth training and inference of the YOLOv8-based detection pipeline, the following system specifications are recommended:
- GPU: NVIDIA Tesla T4 (or better) with at least 8 GB VRAM
- RAM: Minimum 16 GB system memory
- CPU: Quad-core processor or higher
- Python: Version ≥ 3.9, < 3.11

The detection pipeline requires the following major Python libraries:
 
- `ultralytics==8.1.25` – YOLOv8 model training and inference  
- `rasterio==1.3.9` – Reading GeoTIFF metadata and image tiling  
- `opencv-python==4.9.0.80` – Image processing and PNG conversion  
- `albumentations==1.3.1` – Data augmentation for training  
- `pyproj==3.6.1` – Coordinate transformation to WGS 84  
- `numpy==1.24.4` – General scientific libraries  
 
All dependencies are listed in `requirements.txt` and can be installed using: 
```
pip install -r requirements.txt
```
#### Additional Resources
 
- [Label Studio](https://labelstud.io) – Annotation tool used for labeling open slurry tanks.
- [OpenStreetMap](https://download.geofabrik.de) – Vector data source for identifying agricultural locations.

If users wish to use Esri imagery, Esri strongly recommends utilizing the official resources listed below to ensure proper licensing compliance and make full use of their optimized deep learning workflows.
- [Use deep learning for feature extraction and classification—Imagery Workflows | Documentation](https://doc.arcgis.com/en/imagery/workflows/resources/using-deep-learning-for-feature-extraction.htm)
- [Use the model—ArcGIS pretrained models | Documentation](https://doc.arcgis.com/en/pretrained-models/latest/imagery/using-hf-zero-shot-classification.htm)


> Acknowledgement
> This project uses YOLOv8 developed and maintained by Ultralytics. Their open-source object detection framework provided an accessible and efficient foundation for training and deploying the slurry tank detection model.
## 4. Disclaimer

River Deep Mountain AI (“RDMAI”) consists of 10 parties. The parties currently participating in RDMAI are listed at the end of this section and they are collectively referred to in these terms as the “consortium”.

This section provides additional context and usage guidance specific to the artificial intelligence models and / or software (the “**Software**”) distributed under the MIT License. It does not modify or override the terms of the MIT License.  In the event of any conflict between this section and the terms of the MIT licence, the terms of the MIT licence shall take precedence.

#### 1. Research and Development Status
The Software has been created as part of a research and development project and reflects a point-in-time snapshot of an evolving project. It is provided without any warranty, representation or commitment of any kind including with regards to title, non-infringement, accuracy, completeness, or performance. The Software is for information purposes only and it is not: (1) intended for production use unless the user accepts full liability for its use of the Software and independently validates that the Software is appropriate for its required use; and / or (2) intended to be the basis of making any decision without independent validation. No party, including any member of the development consortium, is obligated to provide updates, maintenance, or support in relation to the Software and / or any associated documentation.
#### 2. Software Knowledge Cutoff
The Software was trained on publicly available data up to April 2025. It may not reflect current scientific understanding, environmental conditions, or regulatory standards. Users are solely responsible for verifying the accuracy, timeliness, and applicability of any outputs.
#### 3. Experimental and Generative Nature
The Software may exhibit limitations, including but not limited to:
 - Inaccurate, incomplete, or misleading outputs; 
 - Embedded biases and / or assumptions in training data;
 - Non-deterministic and / or unexpected behaviour;
 - Limited transparency in model logic or decision-making
 
Users must critically evaluate and independently validate all outputs and exercise independent scientific, legal, and technical judgment when using the Software and / or any outputs. The Software is not a substitute for professional expertise and / or regulatory compliance.
 
#### 4. Usage Considerations
 
 - Bias and Fairness: The Software may reflect biases present in its training data. Users are responsible for identifying and mitigating such biases in their applications.
 - Ethical and Lawful Use: The Software is intended solely for lawful, ethical, and development purposes. It must not be used in any way that could result in harm to individuals, communities, and / or the environment, or in any way that violates applicable laws and / or regulations.
 - Data Privacy: The Software was trained on publicly available datasets. Users must ensure compliance with all applicable data privacy laws and licensing terms when using the Software in any way.
 - Environmental and Regulatory Risk: Users are not permitted to use the Software for environmental monitoring, regulatory reporting, or decision making in relation to public health, public policy and / or commercial matters. Any such use is in violation of these terms and at the user’s sole risk and discretion.
 
#### 5. No Liability
 
This section is intended to clarify, and not to limit or modify, the disclaimer of warranties and limitation of liability already provided under the MIT License.
 
To the extent permitted by applicable law, users acknowledge and agree that:
 - The Software is not permitted for use in environmental monitoring, regulatory compliance, or decision making in relation to public health, public policy and / or commercial matters.
 - Any use of the Software in such contexts is in violation of these terms and undertaken entirely at the user’s own risk.
 - The development consortium and all consortium members, contributors and their affiliates expressly disclaim any responsibility or liability for any use of the Software including (but not limited to):
   - Environmental, ecological, public health, public policy or commercial outcomes
   - Regulatory and / or legal compliance failures
   - Misinterpretation, misuse, or reliance on the Software’s outputs
   - Any direct, indirect, incidental, or consequential damages arising from use of the Software including (but not limited to) any (1) loss of profit, (2) loss of use, (3) loss of income, (4) loss of production or accruals, (5) loss of anticipated savings, (6) loss of business or contracts, (7) loss or depletion of goodwill, (8) loss of goods, (9) loss or corruption of data, information, or software, (10) pure economic loss, or (11) wasted expenditure resulting from use of the Software —whether arising in contract, tort, or otherwise, even if foreseeable . 
 
Users assume full responsibility for their use of the Software, validating the Software’s outputs and for any decisions and / or actions taken based on their use of the Software and / or its outputs.

#### 6. Consortium Members  

1. Northumbrian Water Limited
2. Cognizant Worldwide Limited
3. Xylem Water Solutions UK Limited
4. Water Research Centre Limited
5. RSK ADAS Limited
6. The Rivers Trust
7. Wessex Water Limited
8. Northern Ireland Water
9. Southwest Water Limited
10. Anglian Water Services Limited


## 5. References
1. Robinson, C., Bradbury, K., & Borsuk, M. E. (2024). Remotely sensed above-ground storage tank dataset for object detection and infrastructure assessment. *Scientific Data, 11*, 67. https://doi.org/10.1038/s41597-023-02780-1 
2. Khalili, B., & Smyth, A. W. (2024). SOD-YOLOv8—Enhancing YOLOv8 for small object detection in aerial imagery and traffic scenes. *Sensors, 24*(19), 6209. https://doi.org/10.3390/s24196209 
3. Wang, X., Qian, H., Xie, L., Wang, X., & Li, B. (2024). Recognition and classification of typical building shapes based on YOLO object detection models. *ISPRS International Journal of Geo-Information, 13*(12), 433. https://doi.org/10.3390/ijgi13120433
4. Ultralytics. (2024). *YOLOv8 model documentation*. https://docs.ultralytics.com/models/yolov8/
