# Alzheimer's Disease Detection using Machine Learning

A machine learning project that detects Alzheimer's disease from structural brain MRI data, using a Random Forest classifier trained on volumetric measurements of 14 subcortical brain regions.

## Overview

- **Data**: 95 subjects (60 with Alzheimer's, 35 healthy controls)
- **Features**: Volumetric measurements of 14 subcortical brain regions, extracted from 3T structural MRI scans using [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki) (FMRIB Software Library)
- **Model**: Random Forest classifier

The 14 regions (left and right) are: Hippocampus, Amygdala, Putamen, Caudate, Pallidum, Thalamus, and Accumbens.

## Results

| Metric | Value |
|---|---|
| Accuracy | 84.2% |
| AUC | 0.976 |

### Most important brain regions (feature importance)

| Rank | Region | Importance |
|---|---|---|
| 1 | Left Hippocampus (L_Hipp) | 22.3% |
| 2 | Right Hippocampus (R_Hipp) | 17.6% |
| 3 | Left Putamen (L_Puta) | 12.6% |
| 4 | Left Amygdala (L_Amyg) | 10.5% |
| 5 | Left Accumbens (L_Accu) | 6.4% |

The hippocampus being the top predictor lines up with clinical findings, since it is known to be affected early in Alzheimer's disease.

## Project structure

```
alzheimer-detection/
├── alzheimer_detection.py     # Training and evaluation script
├── brain_regions_data.csv     # Volumetric dataset
├── alzheimer_model.pkl        # Trained Random Forest model
├── scaler.pkl                 # Feature scaler
├── structure_names.pkl        # Brain region names
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
python alzheimer_detection.py
```

## Author

Elham Latif

## License

This project is open source and available under the [MIT License](LICENSE).
