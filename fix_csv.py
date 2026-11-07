"""Finds the real dataset on your computer and puts it in this folder.
Run:  python fix_csv.py
"""
import os
import glob
import shutil
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(BASE, "heart_disease_dataset.csv")
NEEDED = ["Age", "Cholesterol", "Blood Pressure", "Heart Rate", "Exercise Hours",
          "Stress Level", "Blood Sugar", "Gender", "Smoking", "Alcohol Intake",
          "Family History", "Diabetes", "Obesity", "Exercise Induced Angina",
          "Chest Pain Type", "Heart Disease"]


def norm(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def is_good(path):
    try:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig",
                         keep_default_na=False, nrows=5)
    except Exception:
        return False
    cols = {norm(c) for c in df.columns}
    return all(norm(n) in cols for n in NEEDED)


if os.path.exists(DEST) and is_good(DEST):
    print("Your dataset is already correct:", DEST)
    raise SystemExit

home = os.path.expanduser("~")
roots = []
for base in (home, os.path.join(home, "OneDrive")):
    for d in ("Downloads", "Desktop", "Documents"):
        roots.append(os.path.join(base, d))

found = []
for r in roots:
    for pattern in ("*.csv", os.path.join("*", "*.csv")):
        for p in glob.glob(os.path.join(r, pattern)):
            if os.path.abspath(p) != os.path.abspath(DEST) and is_good(p):
                found.append(p)

if not found:
    print("Could not find the real dataset on this computer.")
    print("Download heart_disease_dataset.csv from the chat, then run this again.")
    raise SystemExit

src = found[0]
if os.path.exists(DEST):
    shutil.copy(DEST, DEST + ".bad_backup")
shutil.copy(src, DEST)
print("Found the real dataset at:", src)
print("Copied it to            :", DEST)
print("Done. Now run:  python train_model.py")