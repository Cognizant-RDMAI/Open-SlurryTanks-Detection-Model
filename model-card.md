## Training Data
 
The model was trained on a combination of two distinct, high-quality datasets — one publicly available and one custom-created for slurry tank detection.
 
 
### 1. Above-Ground Storage Tank (AGST) Dataset
 
- Sourced from the Microsoft Planetary Computer's NAIP archive

- 27,639 pre-annotated 512×512 JPG images from 2132 NAIP tiles across 48 U.S. states

- Annotations follow PASCAL VOC format, representing multiple tank types:
 
  - closed_roof_tank
 
  - external_floating_roof_tank
 
  - narrow_closed_roof_tank
 
  - sedimentation_tank
 
  - spherical_tank
 
  - undefined_object
 
  - water_tower

- Images and annotations are publicly available and technically validated
 

 
### 2. Custom Annotated Slurry Tank Imagery
 
To specifically enhance the model's performance on slurry tanks:
 
####  Location Identification

- Slurry tank locations were extracted using OpenStreetMap datasets for **England**, **Denmark**, and **Wales**

- Coordinates guided focused imagery collection in rural and agricultural regions
 
#### Image Acquisition

- High-resolution satellite/aerial imagery were used
  
- Used imagery having clear visual distinction of open type circular slurry tank structures
 
#### Manual Annotation

- Slurry tanks were manually annotated using **Label Studio**

- Annotations were exported directly in **YOLO format**, aligned with training input requirements
 
#### Dataset Augmentation

- Due to a limited number of raw images, multiple **data augmentation techniques** were used:

  - Horizontal/vertical flip

  - Color shifting and noise

  - Scaling and cropping

- **Image mosaics** (tile combinations) were introduced to simulate scene complexity and spatial variation

- This enhanced the model’s ability to detect tanks in diverse environmental contexts
 
Together, the AGST dataset and the custom slurry tank dataset ensured a balance between general industrial structures and specific agricultural applications, improving both **robustness** and **domain relevance**.

### Class Labels (8 total)

| ID | Label                        | Source            |
|----|------------------------------|-------------------|
| 0  | closed_roof_tank             | AGST dataset      |
| 1  | external_floating_roof_tank  | AGST dataset      |
| 2  | narrow_closed_roof_tank      | AGST dataset      |
| 3  | sedimentation_tank           | AGST dataset      |
| 4  | spherical_tank               | AGST dataset      |
| 5  | undefined_object             | AGST dataset      |
| 6  | water_tower                  | AGST dataset      |
| 7  | slurry_tank                  | Custom annotation |
 
## Training Configuration
 
- **Model:** YOLOv8l

- **Epochs:** 300

- **Batch Size:** 8

- **Image Size:** 640×640

- **Optimizer:** AdamW

- **Initial LR:** 0.001 → **Final LR:** 0.0001

- **Weight Decay:** 0.0005

- **Patience:** 20 (early stopping)
 
### Data Augmentation

- Mosaic: 1.0

- MixUp: 0.1

- Flip (horizontal/vertical): 0.5 / 0.1

- HSV: (H=0.015, S=0.7, V=0.4)

- Translate: 0.1

- Scale: 0.5

- Dropout: 0.05

- Multi-scale training: yes
 
### Training Platform

- Trained on GPU (device=0) using Ultralytics YOLOv8

- `runs/train/` contains training logs and weights

## Evaluation Metrics

| Class                        | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|------------------------------|-----------|--------|---------|--------------|
| closed_roof_tank             | 0.819     | 0.867  | 0.906   | 0.689        |
| external_floating_roof_tank  | 0.815     | 0.899  | 0.926   | 0.779        |
| narrow_closed_roof_tank      | 0.774     | 0.070  | 0.228   | 0.089        |
| sedimentation_tank           | 0.774     | 0.874  | 0.891   | 0.725        |
| spherical_tank               | 0.760     | 0.654  | 0.691   | 0.483        |
| undefined_object             | 0.278     | 0.145  | 0.129   | 0.110        |
| water_tower                  | 0.719     | 0.730  | 0.762   | 0.533        |
| **slurry_tank**              | **0.945** | **0.778** | **0.883** | **0.737**    |

> **Note:** Slurry tanks achieved the highest precision among all classes, likely due to targeted data collection.

 
## Inference & Post-Processing Workflow
 
1. **Tiling:** GeoTIFFs are automatically split into 512×512 tiles

2. **PNG Conversion:** Tiles are contrast-enhanced and saved as `.png` images

3. **YOLOv8 Inference:** Model predicts object classes + bounding boxes on PNGs

4. **GeoJSON Export:**

   - Pixel coordinates are georeferenced using the source `.tif` transform

   - Results are reprojected to **EPSG:4326 (WGS84)**

   - Final output is a GIS-ready `.geojson` file with class, confidence, and image info
 
> All steps are handled automatically via the processing pipeline script.
 
 
## Strengths

- Performs well on circular agricultural tanks

- Fully geospatially aware output (compatible with QGIS, ArcGIS, etc.)

- Lightweight and GPU-optimized
 
## Limitations

- May misclassify small or partially occluded tanks

- Trained only on RGB daylight imagery (not SAR or night)

- Undefined circular structures may trigger false positives
 
 
## Model File

- Filename: `best.pt`

- Format: YOLOv8 / PyTorch

- Size: 83.6mb
 
## Usage Example
 
```python
from ultralytics import YOLO
 
model = YOLO("path/to/best.pt")
results = model("path/to/image.jpg")
results.show()
```
  


 
 
