import os
import sys
import pickle
import subprocess
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Always find files next to this script, no matter where you run it from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "heart_model.pkl")

# If the model file is missing, train it automatically (runs train_model.py)
if not os.path.exists(MODEL_PATH):
    print("heart_model.pkl not found. Training the model first...")
    result = subprocess.run([sys.executable, os.path.join(BASE_DIR, "train_model.py")])
    if result.returncode != 0:
        print("\nTraining failed. Read the error printed ABOVE this line,")
        print("or run:  python check_setup.py")
        sys.exit(1)

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
NUM_COLS = bundle["num_cols"]
CAT_COLS = bundle["cat_cols"]
NUM_RANGES = bundle["num_ranges"]      # min/max seen in the training data
CAT_OPTIONS = bundle["cat_options"]


# ---------------------------------------------------------------------------
# Numeric feature info: label, unit, description, normal range, input limits.
# General reference ranges only (not medical advice).
# ---------------------------------------------------------------------------
NUM_INFO = {
    "Age": dict(
        label="Age", unit="years", min=1, max=120, step=1, default=50,
        desc="Patient age. Heart disease risk rises with age.",
        normal="Risk rises after about 45 (men) / 55 (women)"),
    "Cholesterol": dict(
        label="Cholesterol", unit="mg/dL", min=50, max=700, step=1, default=200,
        desc="Total cholesterol in the blood. High levels can clog arteries.",
        normal="Below 200 desirable | 200-239 borderline | 240+ high"),
    "Blood Pressure": dict(
        label="Blood pressure (systolic)", unit="mmHg", min=50, max=260, step=1, default=120,
        desc="Pressure in the arteries when the heart beats (top number).",
        normal="Below 120 normal | 120-129 elevated | 130+ high"),
    "Heart Rate": dict(
        label="Heart rate", unit="bpm", min=30, max=220, step=1, default=75,
        desc="Heart beats per minute at rest.",
        normal="60-100 bpm at rest"),
    "Blood Sugar": dict(
        label="Blood sugar", unit="mg/dL", min=30, max=600, step=1, default=100,
        desc="Glucose level in the blood (treated here as a fasting reading).",
        normal="70-99 normal | 100-125 prediabetes range | 126+ diabetes range"),
    "Exercise Hours": dict(
        label="Exercise", unit="hours per week", min=0, max=40, step=1, default=4,
        desc="Hours of physical activity per week.",
        normal="At least 2.5 hours per week recommended"),
    "Stress Level": dict(
        label="Stress level", unit="scale 1-10", min=1, max=10, step=1, default=5,
        desc="Self-reported stress, 1 = very low, 10 = very high.",
        normal="1-3 low | 4-6 moderate | 7-10 high"),
}


def assess_numeric(col, v):
    """Return (status, note) for one numeric value. Status: low/normal/borderline/high/info."""
    if col == "Age":
        return "info", "Older age raises risk" if v >= 55 else "Lower-risk age band"
    if col == "Cholesterol":
        if v < 200: return "normal", "Desirable"
        if v < 240: return "borderline", "Borderline high"
        return "high", "High"
    if col == "Blood Pressure":
        if v < 90:  return "low", "Low"
        if v < 120: return "normal", "Normal"
        if v < 130: return "borderline", "Elevated"
        return "high", "High"
    if col == "Heart Rate":
        if v < 60:  return "low", "Below normal resting range"
        if v <= 100: return "normal", "Normal"
        return "high", "Above normal resting range"
    if col == "Blood Sugar":
        if v < 70:  return "low", "Low"
        if v < 100: return "normal", "Normal"
        if v < 126: return "borderline", "Prediabetes range"
        return "high", "Diabetes range"
    if col == "Exercise Hours":
        if v < 2.5: return "low", "Below recommended 2.5 h/week"
        return "normal", "Meets recommendation"
    if col == "Stress Level":
        if v <= 3: return "normal", "Low stress"
        if v <= 6: return "borderline", "Moderate stress"
        return "high", "High stress"
    return "info", ""


# ---------------------------------------------------------------------------
# Categorical feature info: label, description, which values are risk factors.
# ---------------------------------------------------------------------------
CAT_INFO = {
    "Gender": dict(label="Gender", desc="Biological sex of the patient.", risk=[]),
    "Smoking": dict(label="Smoking", desc="Smoking status: Current, Former or Never.",
                    risk=["Current"]),
    "Alcohol Intake": dict(label="Alcohol intake",
                           desc="How much alcohol the patient drinks: None, Moderate or Heavy.",
                           risk=["Heavy"]),
    "Family History": dict(label="Family history of heart disease",
                           desc="Whether a close relative has had heart disease.", risk=["Yes"]),
    "Diabetes": dict(label="Diabetes", desc="Whether the patient has diabetes.", risk=["Yes"]),
    "Obesity": dict(label="Obesity", desc="Whether the patient is obese.", risk=["Yes"]),
    "Exercise Induced Angina": dict(label="Exercise-induced angina",
                                    desc="Chest pain that appears during physical effort.",
                                    risk=["Yes"]),
    "Chest Pain Type": dict(label="Chest pain type",
                            desc="Typical: classic heart-related pain. Atypical: some typical features. "
                                 "Non-anginal: not heart-related. Asymptomatic: no chest pain.",
                            risk=[]),
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/schema")
def schema():
    """Everything the frontend needs to build the form."""
    num = {}
    for c in NUM_COLS:
        num[c] = dict(NUM_INFO[c],
                      dataset_min=NUM_RANGES[c][0], dataset_max=NUM_RANGES[c][1])
    cat = {c: dict(CAT_INFO[c], options=CAT_OPTIONS[c]) for c in CAT_COLS}
    return jsonify(numeric=num, categorical=cat)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    row = {}

    for col in NUM_COLS:
        info = NUM_INFO[col]
        if col not in data or str(data[col]).strip() == "":
            return jsonify(error=f"Missing value for '{info['label']}'"), 400
        try:
            value = float(data[col])
        except (TypeError, ValueError):
            return jsonify(error=f"'{info['label']}' must be a number"), 400
        if not info["min"] <= value <= info["max"]:
            return jsonify(error=f"'{info['label']}' must be between "
                                 f"{info['min']} and {info['max']} {info['unit']}"), 400
        row[col] = value

    for col in CAT_COLS:
        value = data.get(col)
        if value not in CAT_OPTIONS[col]:
            return jsonify(error=f"'{CAT_INFO[col]['label']}' must be one of {CAT_OPTIONS[col]}"), 400
        row[col] = value

    X = pd.DataFrame([row], columns=NUM_COLS + CAT_COLS)
    prob = float(model.predict_proba(X)[0][1])
    level = "Low" if prob < 0.33 else "Moderate" if prob < 0.66 else "High"

    # Per-feature report: your value vs the reference range
    numeric_report = []
    for col in NUM_COLS:
        status, note = assess_numeric(col, row[col])
        numeric_report.append(dict(
            name=col, label=NUM_INFO[col]["label"], value=row[col],
            unit=NUM_INFO[col]["unit"], normal=NUM_INFO[col]["normal"],
            status=status, note=note))

    risk_factors = [CAT_INFO[c]["label"] + ": " + row[c]
                    for c in CAT_COLS if row[c] in CAT_INFO[c]["risk"]]

    return jsonify(prediction=int(prob >= 0.5), probability=round(prob * 100, 1),
                   level=level, numeric_report=numeric_report,
                   risk_factors=risk_factors, chest_pain=row["Chest Pain Type"])


if __name__ == "__main__":
    app.run(debug=True)