"""Prints a setup report. Run:  python check_setup.py"""
import os
import sys
import importlib

BASE = os.path.dirname(os.path.abspath(__file__))
print("Python  :", sys.version.split()[0])
print("Location:", sys.executable)
print("Folder  :", BASE)
print()

for pkg in ["flask", "pandas", "sklearn"]:
    try:
        m = importlib.import_module(pkg)
        print("OK      ", pkg, getattr(m, "__version__", ""))
    except ImportError:
        print("MISSING ", pkg, " -> run: python -m pip install flask pandas scikit-learn")

print()
for f in ["app.py", "train_model.py", "heart_disease_dataset.csv",
          os.path.join("templates", "index.html")]:
    print("OK      " if os.path.exists(os.path.join(BASE, f)) else "MISSING ", f)

csv = os.path.join(BASE, "heart_disease_dataset.csv")
if os.path.exists(csv):
    try:
        import pandas as pd
        df = pd.read_csv(csv, sep=None, engine="python",
                         encoding="utf-8-sig", keep_default_na=False)
        print("\nCSV shape  :", df.shape, "(correct file: (1000, 16))")
        print("CSV columns:", list(df.columns))
    except Exception as e:
        print("\nCould not read the CSV:", e)

pkl = os.path.join(BASE, "heart_model.pkl")
print("\nheart_model.pkl:", "exists" if os.path.exists(pkl) else "not created yet")