"""Shared data loading, preprocessing and evaluation helpers for ML Assignment 2."""
from pathlib import Path
import urllib.request
import zipfile
import io

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix,
)

SEED = 2026
TEST_SIZE = 0.20

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model" / "saved_models"
RESULTS_DIR = ROOT / "results"
TEST_CSV = ROOT / "test_data.csv"

UCI_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]
TARGET = "income"
POSITIVE = ">50K"

NUMERIC_FEATURES = [
    "age", "fnlwgt", "education_num", "capital_gain", "capital_loss",
    "hours_per_week",
]
CATEGORICAL_FEATURES = [
    "workclass", "marital_status", "occupation", "relationship", "race",
    "sex", "native_country",
]
# 'education' is dropped: it is a 1-to-1 text version of 'education_num'.
DROPPED_FEATURES = ["education"]


def _ensure_raw_files():
    DATA_DIR.mkdir(exist_ok=True)
    needed = [DATA_DIR / "adult.data", DATA_DIR / "adult.test"]
    if all(p.exists() for p in needed):
        return
    with urllib.request.urlopen(UCI_URL) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    for name in ("adult.data", "adult.test", "adult.names"):
        (DATA_DIR / name).write_bytes(archive.read(name))


def load_raw_data():
    """Load the full UCI Adult dataset (adult.data + adult.test = 48,842 rows).

    The '?' placeholders used by UCI are converted to real NaN values.
    """
    _ensure_raw_files()
    read = dict(names=COLUMNS, skipinitialspace=True, na_values="?")
    part_a = pd.read_csv(DATA_DIR / "adult.data", **read)
    part_b = pd.read_csv(DATA_DIR / "adult.test", skiprows=1, **read)
    df = pd.concat([part_a, part_b], ignore_index=True)
    # adult.test labels carry a trailing dot ('>50K.'), normalise them.
    df[TARGET] = df[TARGET].str.strip().str.rstrip(".")
    return df


def clean_data(df):
    """Drop exact duplicate rows and redundant columns. Returns (clean_df, report)."""
    report = {"rows_before": len(df), "duplicates": int(df.duplicated().sum())}
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.drop(columns=DROPPED_FEATURES)
    report["rows_after"] = len(df)
    return df, report


def split_data(df):
    """Stratified 80:20 split with a fixed seed. Target is encoded 1 = '>50K'."""
    X = df.drop(columns=[TARGET])
    y = (df[TARGET] == POSITIVE).astype(int)
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )


def build_preprocessor(scale=True):
    """ColumnTransformer that is fitted on training data only (inside a Pipeline).

    numeric     -> median imputation (+ standard scaling when scale=True)
    categorical -> most-frequent imputation + one-hot encoding
    """
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=10, sparse_output=False)),
    ]
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC_FEATURES),
        ("cat", Pipeline(cat_steps), CATEGORICAL_FEATURES),
    ])


def compute_metrics(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_score)
    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm, y_score
