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

## Model Validation and Performance
The Slurry Tank Detection Model was evaluated through complementary validation strategies to ensure robustness, geographic generalization, and real-world reliability.

### Augmentation-Based Validation
 
The AGST dataset was combined with the annotated slurry tank images.The presence of these non-slurry circular structures makes the dataset valuable for assessing the false positives, where the model might confuse similar looking structures for slurry tanks. This combined dataset design ensured that the validation reflected realistic challenges in distinguishing slurry tanks from visually similar objects in imagery. 

Using the **Albumentations** library, a series of realistic augmentations were applied, such as horizontal and vertical flips, rotations, brightness/contrast adjustments, hue and saturation shifts, Gaussian noise, motion blur, and JPEG compression, to mimic real-world variations in aerial imagery.  
 
Each augmentation was **stochastic**, meaning that every run produced slightly different transformations, with parameters randomly sampled from realistic sensor and environmental ranges. **Bounding boxes were automatically adjusted** after each geometric transformation to maintain accurate labels.  

| **Category** | **Operation (Albumentations)** | **Key Parameters** | **Probability (p)** | **Notes** |
|---------------|--------------------------------|--------------------|---------------------|------------|
| **Geometric** | HorizontalFlip | — | 0.5 | Applied independently |
|  | VerticalFlip | — | 0.5 | — |
|  | Affine | scale = (0.9, 1.1), translate_percent = (0.05, 0.05), rotate = (-15°, 15°) | 0.5 | Small shifts / rotations |
| **Photometric** | RandomBrightnessContrast | — | 0.3 | Simulates lighting variation |
|  | HueSaturationValue | hue ± 10, sat ± 15, val ± 10 | 0.3 | Colour variation |
| **Artefacts** | GaussNoise | var_limit = (10, 50) | 0.2 | Sensor noise |
|  | MotionBlur | blur_limit = 5 | 0.2 | Blur / stitching effect |
|  | ImageCompression | quality = 50–100 | 0.2 | Compression artefacts |


This procedure was repeated multiple times with different random seeds, enabling the model to be evaluated under a wide range of visual conditions without requiring additional ground-truth data. The approach provided a robust assessment of the model’s performance and resilience to variations in lighting, orientation, and image quality.

#### Augmentation-Based Validation Results
 
The trained YOLOv8 model was evaluated across **four independent augmentation runs**, each using different random seeds to generate stochastic variants of the dataset. This approach ensured a balanced and realistic assessment of model robustness under varying conditions.
 
| **Iteration** | **Precision** | **Recall** | **mAP@50** | **Slurry Tank Count** |
|----------------|---------------|-------------|-------------|------------------------|
| 1 | 0.934 | 0.821 | 0.890 | 806 |
| 2 | 0.929 | 0.780 | 0.868 | 809 |
| 3 | 0.939 | 0.830 | 0.896 | 808 |
| 4 | 0.949 | 0.802 | 0.867 | 807 |
| **Average** | **0.937** | **0.808** | **0.880** | — |
 
**Interpretation:**  
The model demonstrates **high precision (~0.94)** and **strong recall (~0.81)** across all runs, confirming its stability under diverse imagery conditions such as lighting variation, orientation changes, sensor noise, and compression artefacts. The consistent mAP@50 score (~0.88) highlights the model’s **robustness and general reliability** even when subjected to multiple randomized augmentation scenarios.

### Farmyard-Level (Geographic Generalization) Validation
 
The aim of this validation strategy was to test how well the **Slurry Tank Detection Model** generalizes to **unseen geographic locations**, especially where background conditions or object similarities could challenge the model.  
 
Unlike augmentation-based validation, which stresses the model using visual variations of the same dataset, this approach directly evaluates model performance in **new environments not used during training**.
 
A diverse set of **geographic locations** across the UK was carefully selected using a manual desktop survey. Each site was chosen to represent specific detection challenges the model might encounter in real-world deployment.  
 
Broadly, the validation was structured around two complementary objectives:  
1. **Detection reliability:** Assessing the model’s ability to correctly detect slurry tanks (True Positives-TP) and identify missed detections (False Negatives-FN).  
2. **False positive control:** Evaluating the model’s tendency to misidentify other circular objects as slurry tanks, capturing False Positives (FP) and True Negatives (TN).  
 
The **farmyard polygons** were obtained from the **OpenStreetMap (OSM)** dataset for the UK. Each polygon was visually inspected through a **manual desktop survey**, and locations were classified into the categories listed below.
 
#### Summary of Instances Considered
 
| **Sl. No.** | **Instance** | **No. of Locations** |
|--------------|---------------|----------------------|
| 1 | Farmyards with slurry tanks | 96 |
| 2 | Farmyards with partially hidden slurry tanks | 13 |
| 3 | Farmyards with circular objects but no slurry tanks | 22 |
| 4 | Farmyards with no slurry tanks | 50 |
| 5 | Circular objects outside farmyards | 23 |
| 6 | Circular objects in agricultural landscapes | 12 |
| 7 | Dense clusters of non-slurry tank circular objects | 17 |

### Farmyard-Level Validation Results
 
From the aggregated confusion matrix (summarized below), the model’s performance metrics were derived using standard evaluation formulas.
 
#### Overall Confusion Matrix
 
| **Class** | **Predicted Slurry Tank** | **Predicted Non-Slurry Tank** |
|------------|----------------------------|-------------------------------|
| **Actual Slurry Tank** | **113 (TP)** | **2 (FN)** |
| **Actual Non-Slurry Tank** | **46 (FP)** | **645 (TN)** |
 
#### Performance Metrics

Formulas used:

- **Precision** = TP / (TP + FP)  

- **Recall** = TP / (TP + FN)  

- **F1 Score** = 2 × (Precision × Recall) / (Precision + Recall)  
 
**Calculated results:**

- **Precision:** 71.0% - Of all objects predicted as slurry tanks, about **7 out of 10 were actual slurry tanks**.  

- **Recall:** 98.26% - The model successfully detected **nearly all true slurry tanks**.  

- **F1 Score:** 82.48% - A balanced measure of detection **reliability and completeness**.  
 
**Interpretation:**  

These results indicate that the model is **highly sensitive** (very unlikely to miss slurry tanks), as shown by the near-perfect recall. The lower precision reflects some **false positives**, primarily in environments with **visually similar circular structures** such as silos or water tanks.  
 
**Note on Accuracy:**  

While accuracy can be computed as:
 
> **Accuracy = (TP + TN) / (TP + FP + FN + TN)**  
 
it is **not meaningful** for object detection tasks like YOLO, since the number of True Negatives (TN) is not well-defined. In this validation, TN was used only in a **controlled sense**, when an entire validation location (e.g., a farmyard or industrial site) contained no slurry tanks, and the model correctly made no detections.
 
**Dataset composition and limitations:**  

The composition of the validation dataset influences the confusion matrix results. As no single catchment with sufficient known slurry tank locations was available, a **structured validation dataset** was created using the **OpenStreetMap (OSM) farmyard layer** and manually verified circular structures. This ensured that the validation remained **systematic and representative**, even though it was not fully catchment-based.

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

- Undefined circular structures might trigger false positives
 
 
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
  
