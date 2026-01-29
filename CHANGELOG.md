# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-01-30

### Added
- Complete crop classification pipeline (6 steps)
- Data exploration with visualization (class distribution, feature analysis, correlation matrix)
- 5 scikit-learn/XGBoost models: MLP, MLP+SMOTE, XGBoost, XGBoost+SMOTE, Random Forest
- 4 PyTorch deep learning models: SpectralMLP, SpectralCNN1D, SpectralHybrid, SpectralAttention
- FocalLoss implementation for class imbalance handling
- SMOTE oversampling integration
- Wall-to-wall satellite image classification (sklearn and PyTorch)
- Band reordering between training data and GeoTIFF
- Google Earth Engine data acquisition script
- Conda (`environment.yml`) and pip (`requirements.txt`) setup
- Makefile for one-command pipeline execution
- GitHub CI workflow (lint + validation)
- CITATION.cff for academic citation
- Contributing guidelines and issue/PR templates
- Full result visualizations in `docs/`
