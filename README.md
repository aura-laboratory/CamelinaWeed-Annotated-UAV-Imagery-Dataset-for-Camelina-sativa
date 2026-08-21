# CamelinaWeed: An Expert-Agronomist-Annotated UAV RGB and Multispectral Dataset for Weed and Crop Monitoring in *Camelina sativa*

<p align="center">
  <b>
    RGB and multispectral UAV imagery with polygon-based weed annotations by expert agronomists for precision agriculture research.
  </b>
</p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.20148697"><b>Dataset</b></a> ·
  <a href="https://doi.org/10.1016/j.dib.2026.113135"><b>Paper</b></a> ·
  <a href="https://doi.org/10.5281/zenodo.20148697"><b>DOI</b></a>
</p>

<p align="center">
  <img src="figures/CamelinaWeed.png" width="470">
</p>

## Table of Contents

- [Overview](#overview)
- [Annotation Preview](#annotation-preview)
- [Specifications Table](#specifications-table)
- [UAV Data Summary](#uav-data-summary)
- [UAV Flight Parameters](#uav-flight-parameters)
- [Dataset Evaluation](#dataset-evaluation)
- [Dataset Structure](#dataset-structure)
- [Data Preparation and Visualization Tools](#data-preparation-and-visualization-tools)
  
## Overview

This repository provides documentation for **CamelinaWeed**, a UAV-based dataset collected in *Camelina sativa* fields in Greece.

The dataset includes RGB and multispectral UAV imagery acquired from agricultural fields in Thessaloniki and Chalkidiki during summer 2025 and winter 2025–2026. It contains annotated RGB images with polygon-based weed annotations created by human experts. In addition to the annotated data, the dataset also provides raw RGB imagery, multispectral images, and orthomosaic products.

CamelinaWeed was created to support research in computer vision, precision agriculture, weed detection, crop monitoring, and field-level analysis under realistic agricultural conditions.


## Annotation Preview

<p align="center">
  <img src="figures/Raw%20UAV-acquired%20image.png" width="45%">
  &nbsp;&nbsp;
  <img src="figures/weed-mapping-annotation.png" width="45%">
</p>

<p align="center">
<b>Turning raw UAV imagery into precise weed intelligence.</b><br>
<i>Expert polygon annotations for field-level weed detection, mapping, and monitoring.</i>
</p>

---

## Specifications Table

| Field | Description |
|---|---|
| **Subject** | Computer Science |
| **Specific subject area** | Computer Vision, Precision Agriculture, Weed Detection |
| **Type of data** | RGB images, annotation masks |
| **How data were acquired** | UAV imagery was acquired using a DJI Phantom 4 Pro and a DJI Mavic 3M. <br><br> DJI Phantom 4 Pro: RGB camera, 1'' CMOS, 20 MP effective pixels, FOV 84°, 8.8 mm / 24 mm equivalent focal length, f/2.8–f/11, autofocus from 1 m to ∞. <br><br> DJI Mavic 3M RGB camera: 4/3 CMOS, 20 MP effective pixels, FOV 84°, 24 mm equivalent focal length, f/2.8–f/11, focus from 1 m to ∞. <br><br> DJI Mavic 3M multispectral camera: 1/2.8-inch CMOS, 5 MP effective pixels, FOV 73.91° (61.2° × 48.10°), 25 mm equivalent focal length, f/2.0, fixed focus. Bands: Green (560 ± 16 nm), Red (650 ± 16 nm), Red Edge (730 ± 16 nm), Near Infrared (860 ± 26 nm). |
| **Data format** | Annotated RGB images: 3023 JPEG images; raw unannotated RGB images: JPEG; polygon annotations: JSON; orthomosaics: GeoTIFF; raw unannotated multispectral images: multi-band TIFF. |
| **Description of data collection** | RGB images were collected while UAVs performed coverage missions over *Camelina sativa* fields. During flights, the camera gimbal was adjusted to -89°, vertically oriented toward the field. Image acquisition was performed at flight altitudes of 2 m, 3 m, 5 m, and 10 m, depending on the field and UAV platform. Flight speed was set to 3 m/s. The dataset includes data from three agricultural fields in Thessaloniki and Chalkidiki, Greece, during summer and winter cultivation periods. |
| **Data source location** | Winter cultivation field — Thessaloniki, Greece: <br> `[40.766551, 22.993202; 40.766327, 22.994137; 40.767043, 22.994564; 40.767380, 22.993470]` <br><br> Winter cultivation field — Chalkidiki, Greece: <br> `[40.368424, 23.068174; 40.368555, 23.069137; 40.369965, 23.068723; 40.369791, 23.067736]` <br><br> Summer cultivation field — Thessaloniki, Greece: <br> `[40.565772, 22.990067; 40.566154, 22.991342; 40.568324, 22.989971; 40.567646, 22.988174; 40.566623, 22.988909]` |
| **Data accessibility** | **Repository name:** *CamelinaWeed--A UAV Dataset for Crop Monitoring, Weed Mapping, and Field Analysis in Camelina sativa* <br><br> **Direct URL to data:** [Zenodo dataset](https://doi.org/10.5281/zenodo.20148697) <br><br> **GitHub repository:** [CamelinaWeed](https://github.com/aura-laboratory/A-UAV-Dataset-for-Crop-Monitoring-Weed-Mapping-and-Field-Analysis-in-Camelina-sativa) <br><br> **DOI:** [10.5281/zenodo.20148697](https://doi.org/10.5281/zenodo.20148697) <br><br> **Database description:** The complete dataset files are hosted on Zenodo, while the GitHub repository provides dataset documentation, annotation information, and data preparation and visualization tools. |

## UAV Data Summary
The following tables summarize the UAV imagery included in the dataset. The data are organized into two groups: **Data for Orthomosaic Generation**, which includes full-field acquisitions used to produce orthomosaic products, and **Data for Weed Detection**, which includes image sets categorized according to the presence or absence of visible weeds.


## Data for Orthomosaic Generation

| Season | Location | Acquisition setting | Images | Orthomosaic |
|:---:|:---:|:---:|:---:|:---:|
| Winter 2025–2026 | Thessaloniki | Mavic 3M flight at 20 m altitude RGB | 227 | ✓ |
| Winter 2025–2026 | Thessaloniki | Mavic 3M flight at 20 m altitude MS | 908 | ✓ |
| Winter 2025–2026 | Chalkidiki | Mavic 3M flight at 20 m altitude RGB | 1351 | ✓ |
| Winter 2025–2026 | Chalkidiki | Mavic 3M flight at 20 m altitude MS | 5404 | ✓ |

## Data for Weed Detection 

| Season | Location | Acquisition setting | Weed-positive images | Weed-negative images |
|:---:|:---:|:---:|:---:|:---:|
| Summer 2025 | Thessaloniki | Phantom flight at 5 m altitude | 34 | 32 |
| Summer 2025 | Thessaloniki | Phantom flight at 10 m altitude | 297 | 46 |
| Winter 2025–2026 | Thessaloniki | Phantom flight at 3 m altitude | 17 | 32 |
| Winter 2025–2026 | Thessaloniki | Mavic 3M flight 1 at 2 m altitude | 627 | 215 |
| Winter 2025–2026 | Thessaloniki | Mavic 3M flight 1 at 2 m altitude MS | — | 842 |
| Winter 2025–2026 | Thessaloniki | Mavic 3M flight 2 at 2 m altitude | 47 | 193 |
| Winter 2025–2026 | Thessaloniki | Mavic 3M flight 2 at 2 m altitude MS | — | 240 |
| Winter 2025–2026 | Chalkidiki | Phantom flight at 3 m altitude | 43 | 159 |
| Winter 2025–2026 | Chalkidiki | Phantom flight at 5 m altitude | 55 | 144 |


## UAV Flight Parameters

The following tables summarize the main flight and imaging parameters used during UAV data acquisition. Parameters are reported separately for **Flight Parameters for Orthomosaic Generation**, where full-field coverage required predefined overlap settings, and **Flight Parameters for Weed Detection**, where low-altitude flights were used to capture detailed crop and weed imagery at different spatial resolutions.

### Flight Parameters for Orthomosaic Generation

| Location | Acquisition setting | Drone | Camera | GSD (cm/pixel) | Frontlap (%) | Sidelap (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Thessaloniki | Mavic 3M flight at 20 m altitude RGB | Mavic 3M | RGB | 0.5 | 85 | 70 |
| Thessaloniki | Mavic 3M flight at 20 m altitude MS | Mavic 3M | MS | 0.5 | 85 | 70 |
| Chalkidiki | Mavic 3M flight at 20 m altitude RGB | Mavic 3M | RGB | 0.5 | 85 | 70 |
| Chalkidiki | Mavic 3M flight at 20 m altitude MS | Mavic 3M | MS | 0.5 | 85 | 70 |

### Flight Parameters for Weed Detection

| Location | Acquisition setting | Drone | Camera | GSD (cm/pixel) |
|:---:|:---:|:---:|:---:|:---:|
| Thessaloniki | Phantom flight at 5 m altitude | Phantom 4 Pro | RGB | 0.14 |
| Thessaloniki | Phantom flight at 10 m altitude | Phantom 4 Pro | RGB | 0.27 |
| Thessaloniki | Phantom flight at 3 m altitude | Phantom 4 Pro | RGB | 0.08 |
| Thessaloniki | Mavic 3M flight 1 at 2 m altitude | Mavic 3M | RGB | 0.15 |
| Thessaloniki | Mavic 3M flight 1 at 2 m altitude MS | Mavic 3M | MS | 0.15 |
| Thessaloniki | Mavic 3M flight 2 at 2 m altitude | Mavic 3M | RGB | 0.15 |
| Thessaloniki | Mavic 3M flight 2 at 2 m altitude MS | Mavic 3M | MS | 0.15 |
| Chalkidiki | Phantom flight at 3 m altitude | Phantom 4 Pro | RGB | 0.08 |
| Chalkidiki | Phantom flight at 5 m altitude | Phantom 4 Pro | RGB | 0.14 |


## Dataset Evaluation

Representative object-detection and instance-segmentation models were evaluated to demonstrate the practical usability of the dataset. A flight-wise split was used to prevent data leakage between overlapping UAV images.
### Object Detection Results

| Model | Class | Precision | Recall | F1-score | AP@50 | AP@50–95 | Processing time (ms/tile) |
|:---|:---|---:|---:|---:|---:|---:|---:|
| RT-DETR-L | Broadleaf | 0.846 | 0.813 | 0.829 | 0.792 | 0.710 | 9.20 |
| RT-DETR-L | Narrowleaf | 0.878 | 0.821 | 0.849 | 0.841 | 0.723 | 9.18 |
| RT-DETR-L | **Macro average** | **0.862** | **0.817** | **0.839** | **0.817** | **0.717** | **9.19** |
| YOLO26m | Broadleaf | 0.809 | 0.701 | 0.751 | 0.771 | 0.630 | 6.31 |
| YOLO26m | Narrowleaf | 0.824 | 0.784 | 0.804 | 0.835 | 0.670 | 6.30 |
| YOLO26m | **Macro average** | **0.817** | **0.743** | **0.777** | **0.803** | **0.650** | **6.30** |
| YOLO26n | Broadleaf | 0.731 | 0.649 | 0.688 | 0.721 | 0.550 | 4.20 |
| YOLO26n | Narrowleaf | 0.811 | 0.724 | 0.765 | 0.774 | 0.590 | 4.21 |
| YOLO26n | **Macro average** | **0.771** | **0.687** | **0.726** | **0.748** | **0.570** | **4.20** |
| Faster R-CNN ResNet-50-FPN | Broadleaf | 0.782 | 0.669 | 0.721 | 0.730 | 0.510 | 19.00 |
| Faster R-CNN ResNet-50-FPN | Narrowleaf | 0.811 | 0.745 | 0.777 | 0.760 | 0.580 | 19.00 |
| Faster R-CNN ResNet-50-FPN | **Macro average** | **0.797** | **0.707** | **0.749** | **0.745** | **0.545** | **19.00** |


### Instance Segmentation Results

| Model | Class | Mask Precision | Mask Recall | Mask F1-score | Mask AP@50 | Mask AP@50–95 | Processing time (ms/tile) |
|:---|:---|---:|---:|---:|---:|---:|---:|
| YOLO26n-seg | Broadleaf | 0.740 | 0.630 | 0.681 | 0.664 | 0.490 | 3.30 |
| YOLO26n-seg | Narrowleaf | 0.790 | 0.680 | 0.731 | 0.748 | 0.510 | 3.30 |
| YOLO26n-seg | **Macro average** | **0.765** | **0.655** | **0.706** | **0.706** | **0.500** | **3.30** |


## Dataset Structure

The dataset is organized hierarchically by acquisition season, location, UAV flight/acquisition setting, and data type.

```text
CamelinaWeed/
├── Summer 2025/
│   └── Thessaloniki/
│       ├── Phantom Flight at 5 m Altitude/
│       │   ├── Annotated/
│       │   │   ├── images/
│       │   │   └── annotations.json
│       │   └── Unannotated/
│       │       └── images/
│       └── ...
│
└── Winter 2025-2026/
    ├── Thessaloniki/
    │   ├── Phantom Flight at 3 m Altitude/
    │   │   ├── Annotated/
    │   │   │   ├── images/
    │   │   │   └── annotations.json
    │   │   └── Unannotated/
    │   │       └── images/
    │   ├── ...
    │   ├── Orthomosaic_RGB.tif
    │   └── Orthomosaic_MS.tif
    │
    └── Chalkidiki/
        ├── Phantom Flight at 3 m Altitude/
        │   ├── Annotated/
        │   │   ├── images/
        │   │   └── annotations.json
        │   └── Unannotated/
        │       └── images/
        ├── ...
        ├── Orthomosaic_RGB.tif
        └── Orthomosaic_MS.tif
```

## Data Preparation and Visualization Tools

The repository includes utility scripts for annotation visualization and dataset preparation for YOLO-based object detection and segmentation workflows.

### 1. Visualize COCO Polygon Annotations

`visualize_annotations.py` visualizes polygon-based COCO annotations on UAV RGB imagery. The script supports displaying annotations interactively or exporting annotated images for inspection and quality control.

Run for one image:

```bash
python scripts/visualize_annotations.py \
  --input "Summer 2025/Thessaloniki/Phantom Flight at 10 m Altitude/Annotated" \
  --image "DJI_0051_JPG.rf.nh87J09No53uRJWaoKVC.JPG" \
  --show-labels \
  --display
```

Run for one image and save the result:

```bash
python scripts/visualize_annotations.py \
  --input "Summer 2025/Thessaloniki/Phantom Flight at 10 m Altitude/Annotated" \
  --image "DJI_0051_JPG.rf.nh87J09No53uRJWaoKVC.JPG" \
  --show-labels \
  --save
```

Run for all images:

```bash
python scripts/visualize_annotations.py \
  --input "Summer 2025/Thessaloniki/Phantom Flight at 10 m Altitude/Annotated" \
  --all \
  --show-labels \
  --save
```

### 2. Prepare and Evaluate Detection and Segmentation Datasets

`prepare_dataset.py` provides a unified workflow for preparing, training, and evaluating weed-detection and instance-segmentation experiments. Users can select one or more acquisition folders and define random, folder-wise, flight-wise, field-wise, season-wise, or manually specified train/validation/test splits.

For object detection, COCO polygon annotations are converted into axis-aligned bounding boxes. For instance segmentation, the original polygon boundaries are retained and converted into YOLO segmentation labels. In both cases, the split is performed before the images are divided into 768 × 768 tiles.

The script also supports configurable sampling of weed-negative training images. Available values include `0`, `0.10`, `0.15`, or `all`. Negative-image sampling is applied only to the training subset, while all available negative images are retained in validation and test subsets for false-positive evaluation.

List the available acquisition folders:

```bash
python prepare_dataset.py list-folders \
  --input "/path/to/CamelinaWeed"
```

Prepare an object-detection dataset:

```bash
python prepare_dataset.py prepare \
  --input "/path/to/CamelinaWeed" \
  --output prepared_detection_dataset \
  --task detection \
  --split-strategy manual \
  --manual-split split_config.json \
  --class-map class_map.json \
  --negative-ratio 0.15 \
  --tile-size 768 \
  --tile-overlap 0.20
```

Train and compare the supported detection models:

```bash
python prepare_dataset.py train \
  --dataset prepared_detection_dataset \
  --output detection_results \
  --models yolo yolo_light fasterrcnn rtdetr \
  --seeds 42 52 62 \
  --group-metrics flight \
  --device 0
```

The supported object-detection models are YOLO, lightweight YOLO, Faster R-CNN, and RT-DETR. Evaluation results include class-wise and macro-averaged precision, recall, F1-score, AP@50, and AP@50:95 for broadleaf and narrowleaf weeds. Results can be grouped by flight, field, or season and summarized across repeated runs using the mean and standard deviation.

Prepare a separate instance-segmentation dataset:

```bash
python prepare_dataset.py prepare \
  --input "/path/to/CamelinaWeed" \
  --output prepared_segmentation_dataset \
  --task segmentation \
  --split-strategy manual \
  --manual-split split_config.json \
  --class-map class_map.json \
  --negative-ratio 0.15 \
  --tile-size 768 \
  --tile-overlap 0.20
```

Train the YOLO instance-segmentation baseline:

```bash
python prepare_dataset.py train \
  --dataset prepared_segmentation_dataset \
  --output segmentation_results \
  --models yolo_seg \
  --seg-weights yolo26n-seg.pt \
  --seg-epochs 200 \
  --seg-batch 4 \
  --seeds 42 52 62 \
  --device 0
```

The segmentation evaluation reports class-wise and macro-averaged mask precision, recall, F1-score, AP@50, and AP@50:95 for broadleaf and narrowleaf weeds.

Detection and segmentation experiments require separate prepared datasets because their annotation formats differ. Random image-level splitting is available only for exploratory analysis. Flight-wise, field-wise, season-wise, or manually defined held-out splits are recommended for reported experiments to prevent data leakage between overlapping UAV images.
