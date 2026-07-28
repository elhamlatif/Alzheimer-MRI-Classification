import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# Brain region names
regions = [
    'R_pall', 'L_pall', 'R_Caud', 'L_Caud',
    'R_Thal', 'L_Thal', 'R_Puta', 'L_Puta',
    'R_Accu', 'L_Accu', 'R_Hipp', 'L_Hipp',
    'R_Amyg', 'L_Amyg'
]

print("=" * 50)
print("NeuroQuantLab - Alzheimer Detection")
print("=" * 50)

# Load data
df = pd.read_csv('brain_regions_data.csv')
print(f"\nTotal samples: {len(df)}")

# Separate features and labels
X = df.drop('ID', axis=1).values
y = []

for idx in df['ID']:
    if '_S_' in str(idx):
        y.append(0)  # Healthy
    else:
        y.append(1)  # Alzheimer's

y = np.array(y)

print(f"Alzheimer's patients: {sum(y)}")
print(f"Healthy controls: {len(y) - sum(y)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\n" + "=" * 50)
print("Model Performance")
print("=" * 50)
print(f"Accuracy: {acc*100:.1f}%")
print(f"AUC: {auc:.3f}")

# Feature importance
importances = model.feature_importances_
feat_df = pd.DataFrame({
    'Region': regions,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\nTop 5 Biomarkers:")
for i, row in feat_df.head(5).iterrows():
    print(f"   {row['Region']:<12} {row['Importance']*100:>5.1f}%")

# Save model files
joblib.dump(model, 'alzheimer_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(regions, 'structure_names.pkl')

print("\nFiles saved:")
print("   - alzheimer_model.pkl")
print("   - scaler.pkl")
print("   - structure_names.pkl")
print("\nDone!")