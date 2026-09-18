Alzheimer's Disease Detection using Machine Learning
This project classifies Alzheimer's disease versus healthy controls using volumetric measurements from structural brain MRI. It relies on volumes from 14 bilateral subcortical regions and a Random Forest classifier.
Overview
The dataset contains 95 subjects: 60 diagnosed with Alzheimer's disease and 35 healthy controls. All scans are 3T structural MRI. Regional volumes were extracted with FSL (FMRIB Software Library) and used as input features for the model.
The 14 regions (left and right) are:

Hippocampus
Amygdala
Putamen
Caudate
Pallidum
Thalamus
Accumbens

Results
Metric     Value
Accuracy   84.2%
AUC        0.976

Most Important Regions
Feature importance from the Random Forest model ranked the regions as follows:
Rank            Region                                  Importance
1         Left Hippocampus(L_Hipp)              22.3%
2         Right Hippocampus (R_Hipp)            17.6%
3           Left Putamen (L_Puta)                    12.6%
4           Left Amygdala (L_Amyg)                10.5%
5         Left Accumbens (L_Accu)                 6.4%

RankRegionImportance1Left Hippocampus (L_Hipp)22.3%2Right Hippocampus (R_Hipp)17.6%3Left Putamen (L_Puta)12.6%4Left Amygdala (L_Amyg)10.5%5Left Accumbens (L_Accu)6.4%
Both the left and right hippocampus contributed the most to the model's decisions. This is consistent with the well-known early involvement of the hippocampus in Alzheimer's disease. These values simply show which features the model relied on most; they do not imply causality and should not be treated as clinical findings.

Project Structure
alzheimer-detection/
├── alzheimer_detection.py     # Training and evaluation script
├── predict.py                 # Inference script
├── data/
│   └── brain_regions_data.csv # Volumetric dataset
├── models/
│   ├── alzheimer_model.pkl    # Trained Random Forest model
│   ├── scaler.pkl             # Feature scaler
│   └── structure_names.pkl    # Brain region names
├── requirements.txt
├── README.md
└── LICENSE

How to Run
Install the dependencies:
Bashpip install -r requirements.txt
Then run the main script:
Bashpython alzheimer_detection.py
Methods
The pipeline consists of three main steps:

1.Subcortical regions are segmented from structural MRI using FSL and their volumes are extracted.
2.These volumes are used as features for a Random Forest classifier.
3.Model performance is evaluated using accuracy and ROC-AUC.
The trained model, scaler, and structure names are saved as pickle files so they can be reused later.

Notes
This is a portfolio project based on MRI-derived volumetric features. The reported accuracy and AUC apply only to this specific dataset and evaluation setup. They should not be interpreted as clinical diagnostic performance.
Feature importance only indicates which regions the model found most useful during classification. It does not mean those regions cause Alzheimer's disease.

Author
Elham Latif

License
MIT License