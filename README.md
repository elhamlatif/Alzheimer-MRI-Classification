 Alzheimer's Disease Detection using Machine Learning

This project looks at brain MRI scans and tries to tell the difference
between people with Alzheimer's disease and healthy controls. It uses the
volumes of 14 subcortical brain regions as input and a Random Forest
classifier to make the call.

Overview

The dataset has 95 subjects: 60 with Alzheimer's disease and 35 healthy
controls. All scans are 3T structural MRI. The regional volumes were
extracted with FSL (FMRIB Software Library) and used as features for the
model.

The 14 regions (left and right) are:

- Hippocampus
- Amygdala
- Putamen
- Caudate
- Pallidum
- Thalamus
- Accumbens

Results

Metric               Value  
-------------------------
 Accuracy         84.2%  
 AUC                0.976  


 Most Important Regions

The Random Forest ranked the regions by how much it relied on them:

 Rank                         Region                              Importance 
---------------------------------------------------------------
 1                  Left Hippocampus (L_Hipp)            22.3%      
 2                 Right Hippocampus (R_Hipp)           17.6%      
 3                  Left Putamen (L_Puta)                    12.6%      
 4                  Left Amygdala (L_Amyg)                10.5%      
 5                 Left Accumbens (L_Accu)                 6.4%       

The left and right hippocampus mattered most to the model. That lines up
with what we already know about Alzheimer's — the hippocampus is one of
the first regions affected. Just to be clear: these numbers only show
which regions the model leaned on the most. They don't mean those regions
*cause* Alzheimer's, and they shouldn't be read as clinical findings.

Project Structure

alzheimer-detection/
├── alzheimer_detection.py # Training and evaluation script
├── predict.py # Inference script
├── data/
│ └── brain_regions_data.csv # Volumetric dataset
├── models/
│ ├── alzheimer_model.pkl # Trained Random Forest model
│ ├── scaler.pkl # Feature scaler
│ └── structure_names.pkl # Brain region names
├── requirements.txt
├── README.md
└── LICENSE

How to Run

First, install the dependencies:

bash
pip install -r requirements.txt

Then train the model:

bash
python alzheimer_detection.py

To make a prediction with the trained model, either edit the example
values inside predict.py or point it at a one-row CSV:

bash
python predict.py
python predict.py --csv data/one_subject.csv

Methods
The pipeline has three main steps:

Subcortical regions are segmented from structural MRI using FSL, and
their volumes are extracted.

Those volumes are used as features for a Random Forest classifier.

The model is evaluated using accuracy and ROC-AUC.

The trained model, the scaler, and the region names are saved as pickle
files so they can be reused later without retraining.

Notes
This is a portfolio project built on MRI-derived volumetric features. The
reported accuracy and AUC apply only to this specific dataset and
evaluation setup. They should not be treated as clinical diagnostic
performance.

Feature importance only tells you which regions the model found most
useful during classification. It does not mean those regions cause
Alzheimer's disease.

Author
Elham Latif

License
MIT License