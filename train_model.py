import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier


# 1. Load Dataset
data = pd.read_csv("xAPI-Edu-Data.csv")

# 2. Create Dropout Target
# L = Likely Dropout → 1
# M/H = Continue → 0
data["dropout"] = data["Class"].apply(lambda x: 1 if x == "L" else 0)

data.drop("Class", axis=1, inplace=True)


# 3. Feature & Target
X = data.drop("dropout", axis=1)
y = data["dropout"]


# 4. Column Types
cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns


# 5. Preprocessing
num_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])


# 6. Model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)


# 7. Pipeline
pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", model)
])


# 8. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 9. Train
pipeline.fit(X_train, y_train)


# 10. Evaluate
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))


# 11. Save Model
joblib.dump(pipeline, "dropout_model.pkl")

print("Model Saved!")
