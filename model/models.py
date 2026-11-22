"""Definitions of the five classifiers and their hyper-parameter search grids."""
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from pipeline_utils import SEED, build_preprocessor

# name -> (estimator, needs_scaling, small grid searched with 5-fold stratified CV)
MODEL_SPECS = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED),
        True,
        {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=SEED),
        False,
        {"clf__max_depth": [6, 8, 10, 12],
         "clf__min_samples_leaf": [10, 30, 60],
         "clf__criterion": ["gini", "entropy"]},
    ),
    "kNN": (
        KNeighborsClassifier(n_jobs=-1),
        True,
        {"clf__n_neighbors": [11, 21, 31],
         "clf__weights": ["uniform", "distance"]},
    ),
    "Naive Bayes": (
        GaussianNB(),
        True,
        {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]},
    ),
    "Random Forest (Ensemble)": (
        RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
        False,
        {"clf__max_depth": [14, 20],
         "clf__min_samples_leaf": [2, 5],
         "clf__max_features": ["sqrt", 0.3]},
    ),
}


def make_pipeline(name):
    estimator, scale, grid = MODEL_SPECS[name]
    pipe = Pipeline([("prep", build_preprocessor(scale=scale)), ("clf", estimator)])
    return pipe, grid
