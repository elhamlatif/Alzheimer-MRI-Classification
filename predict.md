What this script does

predict.py takes the volumes of 14 subcortical brain regions and runs them through a trained Random Forest model. It outputs a predicted class (control or Alzheimer's) and the model's estimated probability that the sample belongs to the Alzheimer's class.

Loading the trained files

At the top, the script loads three files that were saved during training:

python
model = joblib.load("models/alzheimer_model.pkl")
scaler = joblib.load("models/scaler.pkl")
structure_names = joblib.load("models/structure_names.pkl")
alzheimer_model.pkl is the trained Random Forest itself.
scaler.pkl is the same StandardScaler that was fit during training - it has to be reused here, not refit, or the input values won't be on the same scale the model learned from.
structure_names.pkl holds the 14 region names in the exact order the model expects.
predict_from_values(volumes)

Takes the 14 region volumes directly and:

checks that exactly 14 values came in,
converts them to a NumPy array,
scales them with the same scaler from training,
runs them through the model,
calls predict_proba for the Alzheimer's-class probability, and predict for the class label itself.

It prints the result and returns the probability.

predict_from_csv(file_path)

For when the volumes are sitting in a CSV instead of being passed in by hand. It reads the file, drops the ID column if there is one, checks that all 14 required region columns are present, and pulls the values out in the same order the model was trained on - one subject per file. Then it just hands those values off to predict_from_values().

Running it directly

If you run predict.py on its own, it tests itself with a set of example volumes hardcoded in the __main__ block. To predict from an actual CSV instead, call predict_from_csv() with the file path.

The 14 regions, in order

The order matters here - it has to match what the model was trained on.

	     Region	                        Description
1-2	R_pall, L_pall	Right and left pallidum
3-4	R_Caud, L_Caud	Right and left caudate
5-6	R_Thal, L_Thal	Right and left thalamus
7-8	R_Puta, L_Puta	Right and left putamen
9-10	R_Accu, L_Accu	Right and left accumbens
11-12	R_Hipp, L_Hipp	Right and left hippocampus
13-14	R_Amyg, L_Amyg	Right and left amygdala