"""Train the heart disease model and save it as heart_model.pkl."""
import os
import sys
import pickle
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "heart_disease_dataset.csv")
TARGET = "Heart Disease"

NUM_COLS = ["Age", "Cholesterol", "Blood Pressure", "Heart Rate",
            "Exercise Hours", "Stress Level", "Blood Sugar"]
CAT_COLS = ["Gender", "Smoking", "Alcohol Intake", "Family History", "Diabetes",
            "Obesity", "Exercise Induced Angina", "Chest Pain Type"]
EXPECTED = NUM_COLS + CAT_COLS + [TARGET]


def norm(name):
    """'Blood Pressure', 'blood_pressure', ' BLOOD PRESSURE ' all become 'bloodpressure'."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def load_data(path):
    # sep=None auto-detects comma / semicolon / tab; utf-8-sig removes a hidden BOM.
    # keep_default_na=False matters: "None" in 'Alcohol Intake' is a real
    # category (no alcohol), not a missing value.
    df = pd.read_csv(path, sep=None, engine="python",
                     encoding="utf-8-sig", keep_default_na=False)
    df.columns = [str(c).strip().strip('"').strip("'") for c in df.columns]

    # Match column names ignoring case, spaces and underscores
    lookup = {norm(c): c for c in df.columns}
    df = df.rename(columns={lookup[norm(n)]: n for n in EXPECTED if norm(n) in lookup})

    missing = [c for c in EXPECTED if c not in df.columns]
    if missing:
        print("\nERROR: your CSV does not have the columns this project expects.")
        print("File read        :", path)
        print("Columns in CSV   :", list(df.columns))
        print("Columns expected :", EXPECTED)
        print("Missing          :", missing)
        print("\nFix: use the original heart_disease_dataset.csv (1000 rows, 16 columns),")
        print("or change NUM_COLS / CAT_COLS / TARGET above to match your file.")
        sys.exit(1)

    # Clean values
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CAT_COLS:
        df[c] = df[c].astype(str).str.strip()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    return df.dropna(subset=NUM_COLS + [TARGET])


df = load_data(CSV_PATH)
print("Loaded", len(df), "rows and", len(df.columns), "columns")

X = df[NUM_COLS + CAT_COLS]
y = df[TARGET].astype(int)

pre = ColumnTransformer([
    ("num", StandardScaler(), NUM_COLS),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
])

model = Pipeline([
    ("pre", pre),
    ("clf", RandomForestClassifier(n_estimators=300, random_state=42)),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
model.fit(X_train, y_train)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("ROC-AUC :", round(roc_auc_score(y_test, proba), 4))
print(classification_report(y_test, pred))

# Retrain on all rows for the final model
model.fit(X, y)

bundle = {
    "model": model,
    "num_cols": NUM_COLS,
    "cat_cols": CAT_COLS,
    "num_ranges": {c: [float(df[c].min()), float(df[c].max())] for c in NUM_COLS},
    "cat_options": {c: sorted(df[c].unique().tolist()) for c in CAT_COLS},
    "sklearn_version": sklearn.__version__,
}
with open(os.path.join(BASE_DIR, "heart_model.pkl"), "wb") as f:
    pickle.dump(bundle, f)
print("Saved heart_model.pkl (scikit-learn", sklearn.__version__ + ")")