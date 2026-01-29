# Crop Classification with Deep Learning

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch)
![Sentinel-2](https://img.shields.io/badge/Sentinel--2-ESA-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

Multi-class crop classification in the **Elgabel Region, Sudan** using Sentinel-2 satellite imagery and multiple machine learning / deep learning models. The pipeline covers data acquisition (Google Earth Engine), exploratory analysis, classical ML training (scikit-learn, XGBoost), PyTorch deep learning, and wall-to-wall satellite image classification.

---

## Study Area

**Elgabel Region, Sudan** — an agricultural zone where the following five crop/land-cover classes are mapped using Sentinel-2 imagery from Q1 2020:

| Class ID | Name   | Color   |
|----------|--------|---------|
| 0        | Cotton | #FF8C00 |
| 1        | Wheat  | #FFD700 |
| 2        | Fallow | #8B4513 |
| 3        | Grass  | #32CD32 |
| 4        | Water  | #0000FF |

---

## Workflow

```
1. Data Acquisition      (Google Earth Engine — Sentinel-2 composite + indices)
        ↓
2. Data Exploration      (01_data_exploration.py)
        ↓
3. Preprocessing & ML    (02_preprocessing_and_model.py)
   Models                  ├─ MLP (class weights)
                           ├─ MLP (SMOTE)
                           ├─ XGBoost (class weights)
                           ├─ XGBoost (SMOTE)
                           └─ Random Forest (balanced)
        ↓
4. Apply ML to Image     (03_apply_to_image.py)
        ↓
5. PyTorch Deep Learning  (04_pytorch_models.py)
   Models                  ├─ SpectralMLP
                           ├─ SpectralCNN1D
                           ├─ SpectralHybrid (CNN + MLP)
                           └─ SpectralAttention (Transformer)
        ↓
6. Apply DL to Image     (05_apply_pytorch_to_image.py)
```

---

## Project Structure

```
crop-classification-deep-learning/
├── README.md
├── LICENSE
├── .gitignore
├── environment.yml
├── fix_jupyter_kernel.bat
├── 01_data_exploration.py
├── 02_preprocessing_and_model.py
├── 03_apply_to_image.py
├── 03_diagnose_bands.py
├── 04_pytorch_models.py
├── 05_apply_pytorch_to_image.py
└── gee/
    └── crop_classification_gee.js
```

---

## Data Description

**24 spectral features** derived from a Sentinel-2 Q1 2020 composite:

| Category | Features |
|----------|----------|
| **Spectral Bands (10)** | B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 |
| **Vegetation Indices (14)** | NDVI, EVI, SAVI, NDRE, GNDVI, NDMI, BSI, MNDWI, LSWI, GCVI, WDRVI, CIgreen, CIrededge, MSAVI |

Training data: ~43,000 labeled point samples across 5 classes (exported from GEE as CSV/GeoJSON).

---

## Model Architectures

### Scikit-learn / XGBoost (Step 02)

| Model | Architecture | Imbalance Strategy |
|-------|-------------|-------------------|
| MLP Neural Network | 24→256→128→64→32→5 | Class weights |
| MLP + SMOTE | 24→512→256→128→64→32→5 | SMOTE oversampling |
| XGBoost | 500 trees, depth=8 | Class weights |
| XGBoost + SMOTE | 500 trees, depth=8 | SMOTE oversampling |
| Random Forest | 500 trees | Balanced class weights |

### PyTorch Deep Learning (Step 04)

| Model | Description | Key Features |
|-------|------------|-------------|
| SpectralMLP | Deep MLP (24→128→256→128→64→5) | BatchNorm, Dropout, FocalLoss |
| SpectralCNN1D | 1D CNN (32→64→128 filters) | Treats bands as 1D signal |
| SpectralHybrid | CNN + MLP fusion | Dual-branch feature extraction |
| SpectralAttention | Transformer-style | CLS token, 4 heads, 2 layers |

All PyTorch models use: FocalLoss (gamma=2), AdamW, ReduceLROnPlateau, early stopping, SMOTE-balanced data.

---

## Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate geodl
```

### 2. Register Jupyter kernel (optional)

```bash
fix_jupyter_kernel.bat
```

### 3. Obtain satellite data

Use the Google Earth Engine script in `gee/crop_classification_gee.js` to export:
- `S2_composite_24bands_2020_Q1.tif` — 24-band Sentinel-2 composite
- `S2_key_indices_2020_Q1.tif` — Key spectral indices
- `crop_training_data_5classes_2020.csv` — Labeled training samples

Place the exported files in the project root directory.

---

## Usage

Run the scripts in order:

```bash
# Step 1: Explore the training data
python 01_data_exploration.py

# Step 2: Train scikit-learn and XGBoost models
python 02_preprocessing_and_model.py

# Step 3: Apply best ML model to full satellite image
python 03_apply_to_image.py

# Step 4: Train PyTorch deep learning models
python 04_pytorch_models.py

# Step 5: Apply best PyTorch model to full satellite image
python 05_apply_pytorch_to_image.py
```

### Utility Scripts

- `03_diagnose_bands.py` — Diagnose band ordering between training data and GeoTIFF

---

## Google Earth Engine

The `gee/crop_classification_gee.js` script handles data acquisition:

1. Defines the study area (Elgabel Region, Sudan)
2. Filters Sentinel-2 Surface Reflectance (Q1 2020, <20% cloud)
3. Computes a cloud-free median composite
4. Calculates 14 spectral indices (NDVI, EVI, SAVI, etc.)
5. Samples training points from labeled polygons
6. Exports the 24-band composite and training CSV to Google Drive

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
