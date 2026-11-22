"""Re-score the saved models on test_data.csv to reproduce the reported metrics.

Usage (from the repository root):
    python model/predict.py
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_utils import TARGET, POSITIVE, MODEL_DIR, TEST_CSV, evaluate_model  # noqa: E402


def main():
    test_df = pd.read_csv(TEST_CSV)
    X_test = test_df.drop(columns=[TARGET])
    y_test = (test_df[TARGET] == POSITIVE).astype(int)
    rows = []
    for path in sorted(MODEL_DIR.glob("*.pkl")):
        metrics, _, _ = evaluate_model(joblib.load(path), X_test, y_test)
        rows.append({"Model file": path.name, **metrics})
    print(pd.DataFrame(rows).set_index("Model file").round(4).to_string())


if __name__ == "__main__":
    main()
