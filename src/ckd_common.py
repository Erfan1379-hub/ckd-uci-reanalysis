"""Shared data loading, cleaning and pipeline construction for the UCI CKD re-analysis.

Every step in this module that is applied outside a scikit-learn Pipeline is rule-based
(it learns nothing from the data). Every step that estimates a parameter is a
transformer that is fitted inside a Pipeline, i.e. on training folds only.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "kidney_disease.csv"

CONT = ["age", "bp", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"]   # 11 continuous
ORD = ["sg", "al", "su"]                                                            # 3 ordinal
BIN = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]          # 10 binary
FEATS = CONT[:2] + ORD + BIN[:4] + CONT[2:] + BIN[4:]   # original column order of the file
LOG_VARS = ["bgr", "bu", "sc", "wc", "pot", "sod"]
BIN_MAP = {"normal": 0, "abnormal": 1, "notpresent": 0, "present": 1,
           "no": 0, "yes": 1, "good": 0, "poor": 1}
# Physiological plausibility limits (rule-based, fixed a priori). Values outside are
# treated as data-entry errors and set to missing.
PLAUSIBLE = {"sod": (100, 200), "pot": (1.5, 15)}

NAMES = {"age": "Age", "bp": "Blood pressure", "sg": "Specific gravity", "al": "Albumin",
         "su": "Sugar", "rbc": "Red blood cells (urine)", "pc": "Pus cells",
         "pcc": "Pus cell clumps", "ba": "Bacteria", "bgr": "Blood glucose",
         "bu": "Blood urea", "sc": "Serum creatinine", "sod": "Sodium", "pot": "Potassium",
         "hemo": "Haemoglobin", "pcv": "Packed cell volume", "wc": "White cell count",
         "rc": "Red cell count", "htn": "Hypertension", "dm": "Diabetes mellitus",
         "cad": "Coronary artery disease", "appet": "Appetite", "pe": "Pedal oedema",
         "ane": "Anaemia"}


def load_data(path=DATA, plausibility=True, return_log=False):
    """Rule-based cleaning. Returns X (400 x 24 float DataFrame), y (0/1 Series)."""
    raw = pd.read_csv(path, dtype=str)
    log = {}
    df = raw.copy()
    for c in df.columns:
        df[c] = df[c].str.strip().replace({"?": np.nan, "": np.nan})
    log["duplicates"] = int(df.drop(columns="id").duplicated().sum())
    for c in CONT + ORD:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in BIN:
        df[c] = df[c].map(BIN_MAP).astype(float)
    if plausibility:
        log["implausible"] = {}
        for c, (lo, hi) in PLAUSIBLE.items():
            bad = (df[c] < lo) | (df[c] > hi)
            log["implausible"][c] = df.loc[bad, ["id", c]].values.tolist()
            df.loc[bad, c] = np.nan
    y = (df["classification"] == "ckd").astype(int).rename("ckd")
    X = df[FEATS].astype(float)
    return (X, y, log) if return_log else (X, y)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip selected columns at [Q1 - k*IQR, Q3 + k*IQR]; limits learned on the data passed to fit."""

    def __init__(self, cols=None, k=3.0):
        self.cols = cols
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        q1, q3 = np.nanpercentile(X[:, self.cols], [25, 75], axis=0)
        self.lo_, self.hi_ = q1 - self.k * (q3 - q1), q3 + self.k * (q3 - q1)
        return self

    def transform(self, X):
        X = np.array(X, float)
        X[:, self.cols] = np.clip(X[:, self.cols], self.lo_, self.hi_)   # NaN stays NaN
        return X


class Log1p(BaseEstimator, TransformerMixin):
    """log(1+x) on selected columns (no learned parameters)."""

    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X, float)
        X[:, self.cols] = np.log1p(np.clip(X[:, self.cols], 0, None))
        return X


def make_imputer(name, seed=SEED, add_indicator=False):
    if name == "median":
        return SimpleImputer(strategy="median", add_indicator=add_indicator)
    if name == "knn":
        return KNNImputer(n_neighbors=5, add_indicator=add_indicator)
    if name == "mice":
        return IterativeImputer(max_iter=15, random_state=seed, sample_posterior=False,
                                add_indicator=add_indicator)
    raise ValueError(name)


def preprocessing(features, imputer="mice", add_indicator=False):
    """Winsorise continuous vars -> log skewed vars -> impute -> standardise."""
    cont = [features.index(c) for c in CONT if c in features]
    logc = [features.index(c) for c in LOG_VARS if c in features]
    steps = []
    if cont:
        steps.append(("winsor", Winsorizer(cont, 3.0)))
    if logc:
        steps.append(("log", Log1p(logc)))
    steps += [("impute", make_imputer(imputer, add_indicator=add_indicator)),
              ("scale", StandardScaler())]
    return steps


def classifiers(seed=SEED):
    svm = SVC(C=3, probability=True, random_state=seed)
    rf = RandomForestClassifier(500, random_state=seed)
    hgb = HistGradientBoostingClassifier(random_state=seed)
    return {
        "Logistic regression": LogisticRegression(max_iter=2000),
        "k-nearest neighbours": KNeighborsClassifier(7),
        "Support vector machine": svm,
        "Random forest": rf,
        "Extremely randomised trees": ExtraTreesClassifier(500, random_state=seed),
        "Gradient boosting": hgb,
        "Stacked ensemble": StackingClassifier(
            estimators=[("svm", SVC(C=3, probability=True, random_state=seed)),
                        ("rf", RandomForestClassifier(500, random_state=seed)),
                        ("hgb", HistGradientBoostingClassifier(random_state=seed))],
            final_estimator=LogisticRegression(max_iter=2000), cv=5),
    }


def make_pipeline(clf, features, imputer="mice", add_indicator=False, extra=()):
    return Pipeline(preprocessing(features, imputer, add_indicator) + list(extra) + [("clf", clf)])


class EnvelopeRule(BaseEstimator, ClassifierMixin):
    """Reference-envelope rule.

    fit: for every variable, record the minimum and maximum observed among the training
    patients WITHOUT CKD (the 'control envelope').
    predict: CKD if ANY observed (non-missing) value of a patient lies outside the envelope;
    missing values are ignored. A continuous score for ROC analysis is the number of
    variables outside the envelope (plus a small term for the size of the excursion).
    No imputation, scaling or other preprocessing is used.
    """

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        ctrl = X[y == 0]
        self.lo_ = np.nanmin(ctrl, axis=0)
        self.hi_ = np.nanmax(ctrl, axis=0)
        span = np.where(self.hi_ - self.lo_ > 0, self.hi_ - self.lo_, 1.0)
        self.span_ = span
        self.classes_ = np.array([0, 1])
        return self

    def _excursions(self, X):
        X = np.asarray(X, float)
        below = np.clip(self.lo_ - X, 0, None) / self.span_
        above = np.clip(X - self.hi_, 0, None) / self.span_
        exc = np.nan_to_num(below + above, nan=0.0)
        return exc

    def decision_function(self, X):
        exc = self._excursions(X)
        n_out = (exc > 0).sum(1)
        return n_out + 1 - 1 / (1 + exc.sum(1))

    def predict(self, X):
        return ((self._excursions(X) > 0).sum(1) > 0).astype(int)

    def predict_proba(self, X):
        s = self.decision_function(X)
        p = 1 - 1 / (1 + s)            # monotone map of the score to [0, 1)
        return np.column_stack([1 - p, p])


class MissingnessOnly(BaseEstimator, TransformerMixin):
    """Replace every variable by its missingness indicator (1 = missing)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.isnan(np.asarray(X, float)).astype(float)
