import pandas as pd
import joblib


# Load model
model = joblib.load("dropout_model.pkl")

# Load data
data = pd.read_csv("xAPI-Edu-Data.csv")

# Save ID
data["student_id"] = data.index

# Remove target
X = data.drop(["Class"], axis=1)


# Predict
probs = model.predict_proba(X)[:,1]
preds = model.predict(X)


# Risk Labels
def label_risk(p):
    if p >= 0.7:
        return "High"
    elif p >= 0.4:
        return "Medium"
    else:
        return "Low"


risk_labels = [label_risk(p) for p in probs]


# Output
result = pd.DataFrame({
    "student_id": data["student_id"],
    "risk_score": probs,
    "risk_label": risk_labels,
    "predicted_dropout": preds
})


result.to_csv("predictions.csv", index=False)

print("Predictions saved!")
