"""All experiments of the UCI CKD re-analysis. Writes tables to ../results/.

Run:  python run_analysis.py            (all experiments; ~45 min on 2 CPU cores)
      python run_analysis.py E3 E5      (selected experiments)
Each experiment writes its own file and is skipped if that file already exists
(delete the file to recompute).
"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, adjusted_rand_score, brier_score_loss,
                             confusion_matrix, f1_score, pairwise_distances, precision_score,
                             recall_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     permutation_test_score, train_test_split)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from ckd_common import (BIN, CONT, FEATS, NAMES, ORD, ROOT, SEED, EnvelopeRule, Log1p,
                        MissingnessOnly, Winsorizer, classifiers, load_data, make_imputer,
                        make_pipeline, preprocessing)

warnings.filterwarnings("ignore")
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
N_JOBS = -1
K, R = 5, 10                              # 5-fold CV repeated 10 times = 50 folds
CV = RepeatedStratifiedKFold(n_splits=K, n_repeats=R, random_state=SEED)
X, y, CLEAN_LOG = load_data(return_log=True)
F = list(X.columns)
ORDER = list(classifiers())               # simplest -> most complex; used to break ties


# ---------------------------------------------------------------------------------------
def metrics(y_true, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "f1": f1_score(y_true, pred, zero_division=0),
            "auc": roc_auc_score(y_true, p) if len(set(y_true)) > 1 else np.nan,
            "brier": brier_score_loss(y_true, p),
            "errors": int(fp + fn)}


def _fold(est, Xd, yd, tr, te, fold):
    m = clone(est).fit(Xd.iloc[tr], yd.iloc[tr])
    p = m.predict_proba(Xd.iloc[te])[:, 1]
    return {"fold": fold, **metrics(yd.iloc[te].values, p)}


def cv_eval(est, Xd=None, yd=None, cv=CV):
    Xd = X if Xd is None else Xd
    yd = y if yd is None else yd
    out = Parallel(n_jobs=N_JOBS)(delayed(_fold)(est, Xd, yd, tr, te, i)
                                  for i, (tr, te) in enumerate(cv.split(Xd, yd)))
    return pd.DataFrame(out)


def corrected_t(d, n_train, n_test):
    """Nadeau-Bengio corrected resampled t-test for J paired resampling estimates d."""
    d = np.asarray(d, float)
    J = len(d)
    v = d.var(ddof=1)
    if v == 0:
        return 0.0 if d.mean() == 0 else np.inf, 1.0 if d.mean() == 0 else 0.0
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * v)
    return t, 2 * stats.t.sf(abs(t), J - 1)


def summary(df, cols=("accuracy", "precision", "recall", "specificity", "f1", "auc", "brier")):
    s = {}
    for c in cols:
        s[c + "_mean"] = df[c].mean()
        s[c + "_sd"] = df[c].std(ddof=1)
    s["errors_total"] = int(df["errors"].sum())
    s["perfect_folds"] = int((df["errors"] == 0).sum())
    s["n_folds"] = len(df)
    return s


def done(name):
    p = RES / name
    if p.exists():
        print(f"[skip] {name} exists")
        return True
    return False


def section(t):
    print("\n" + "=" * 80 + f"\n{t}\n" + "=" * 80, flush=True)


# ---------------------------------------------------------------------------------------
def E0_data_audit():
    """Descriptive audit: missingness, control envelope, binary findings in controls."""
    if done("E0_audit.json"):
        return
    section("E0 data audit")
    Xr, yr, log = load_data(plausibility=False, return_log=True)
    out = {"n": len(X), "n_ckd": int(y.sum()), "n_ctrl": int((y == 0).sum()),
           "duplicates": CLEAN_LOG["duplicates"], "implausible_set_missing": CLEAN_LOG["implausible"],
           "missing_cells_pct_raw": float(Xr.isna().mean().mean() * 100),
           "missing_cells_pct_clean": float(X.isna().mean().mean() * 100),
           "rows_with_missing": int(X.isna().any(axis=1).sum()),
           "rows_with_missing_by_class": X.isna().any(axis=1).groupby(y).mean().to_dict(),
           "mean_missing_per_row_by_class": X.isna().sum(axis=1).groupby(y).mean().to_dict()}
    rows = []
    for c in FEATS:
        a, b = X.loc[y == 0, c].dropna(), X.loc[y == 1, c].dropna()
        outside = ((b < a.min()) | (b > a.max()))
        rows.append({"var": c, "name": NAMES[c],
                     "type": "continuous" if c in CONT else "ordinal" if c in ORD else "binary",
                     "missing_pct": X[c].isna().mean() * 100,
                     "missing_pct_ctrl": X.loc[y == 0, c].isna().mean() * 100,
                     "missing_pct_ckd": X.loc[y == 1, c].isna().mean() * 100,
                     "ctrl_min": a.min(), "ctrl_max": a.max(), "ctrl_n": len(a),
                     "ckd_min": b.min(), "ckd_max": b.max(), "ckd_n": len(b),
                     "ckd_outside_ctrl_range_n": int(outside.sum()),
                     "ckd_outside_ctrl_range_pct_observed": outside.mean() * 100,
                     "positive_in_ctrl": int((a == 1).sum()) if c in BIN else np.nan,
                     "positive_in_ckd": int((b == 1).sum()) if c in BIN else np.nan})
    pd.DataFrame(rows).to_csv(RES / "E0_variables.csv", index=False)
    env = EnvelopeRule().fit(X, y)
    exc = env._excursions(X) > 0
    n_out = exc.sum(1)
    out["ckd_with_no_excursion_full_data"] = int(((n_out == 0) & (y == 1)).sum())
    out["ckd_excursions_median"] = float(np.median(n_out[y == 1]))
    out["ckd_excursions_iqr"] = [float(np.percentile(n_out[y == 1], 25)),
                                 float(np.percentile(n_out[y == 1], 75))]
    hard = X[(n_out == 0) & (y == 1)]
    hard.to_csv(RES / "E0_ckd_inside_envelope.csv")
    # pairwise correlations (pairwise complete) above 0.75
    corr = X.corr()
    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(FEATS) for b in FEATS[i + 1:]
             if abs(corr.loc[a, b]) > 0.75]
    out["corr_pairs_gt_0.75"] = pairs
    q1, q3 = X[CONT].quantile(.25), X[CONT].quantile(.75)
    out["iqr15_outliers"] = ((X[CONT] < q1 - 1.5 * (q3 - q1)) | (X[CONT] > q3 + 1.5 * (q3 - q1))).sum().to_dict()
    corr.to_csv(RES / "E0_corr.csv")
    json.dump(out, open(RES / "E0_audit.json", "w"), indent=2, default=float)
    print(json.dumps(out, indent=1, default=float))


# ---------------------------------------------------------------------------------------
def E1_baselines():
    """Rule and trivial baselines under the same 5x10 repeated CV."""
    if done("E1_baselines.csv"):
        return
    section("E1 baselines (envelope rule, shallow trees, missingness-only)")
    simple_imp = [("impute", SimpleImputer(strategy="median"))]
    ests = {
        "Envelope rule": EnvelopeRule(),
        "Decision stump (depth 1)": Pipeline(simple_imp + [("clf", DecisionTreeClassifier(max_depth=1, random_state=SEED))]),
        "Decision tree (depth 2)": Pipeline(simple_imp + [("clf", DecisionTreeClassifier(max_depth=2, random_state=SEED))]),
        "Missingness indicators only (LR)": Pipeline([("miss", MissingnessOnly()), ("clf", LogisticRegression(max_iter=2000))]),
    }
    rows, folds = [], []
    for nm, est in ests.items():
        d = cv_eval(est)
        d["model"] = nm
        folds.append(d)
        rows.append({"model": nm, **summary(d)})
        print(f"{nm:36s} acc={d.accuracy.mean():.4f}  f1={d.f1.mean():.4f}  auc={d.auc.mean():.4f}  "
              f"spec={d.specificity.mean():.4f} sens={d.recall.mean():.4f}", flush=True)
    pd.DataFrame(rows).to_csv(RES / "E1_baselines.csv", index=False)
    pd.concat(folds).to_csv(RES / "E1_baselines_folds.csv", index=False)
    # describe the trees fitted on all data (for the text)
    with open(RES / "E1_trees.txt", "w") as f:
        for nm in ["Decision stump (depth 1)", "Decision tree (depth 2)"]:
            m = clone(ests[nm]).fit(X, y)
            f.write(nm + "\n" + export_text(m[-1], feature_names=F) + "\n")
        mo = clone(ests["Missingness indicators only (LR)"]).fit(X, y)
        coef = pd.Series(mo[-1].coef_[0], index=F).sort_values()
        f.write("Missingness LR coefficients (all data):\n" + coef.round(3).to_string() + "\n")


# ---------------------------------------------------------------------------------------
def E2_classifiers():
    """Seven classifiers, MICE pipeline, 5x10 repeated CV on all 400 patients."""
    if done("E2_classifiers.csv"):
        return
    section("E2 classifiers (repeated CV on all 400)")
    rows, folds = [], []
    for nm, clf in classifiers().items():
        t = time.time()
        d = cv_eval(make_pipeline(clf, F))
        d["model"] = nm
        folds.append(d)
        rows.append({"model": nm, **summary(d)})
        print(f"{nm:28s} acc={d.accuracy.mean():.4f} f1={d.f1.mean():.4f}+/-{d.f1.std():.4f} "
              f"auc={d.auc.mean():.4f} brier={d.brier.mean():.4f} errors={d.errors.sum()} "
              f"({time.time()-t:.0f}s)", flush=True)
    folds = pd.concat(folds)
    env = pd.read_csv(RES / "E1_baselines_folds.csv").query("model == 'Envelope rule'")
    allf = pd.concat([folds, env])
    allf.to_csv(RES / "E2_classifiers_folds.csv", index=False)
    # corrected resampled t-tests vs logistic regression (accuracy and Brier)
    n_te = len(X) / K
    n_tr = len(X) - n_te
    ref = allf[allf.model == "Logistic regression"].sort_values("fold")
    tests = []
    for nm in allf.model.unique():
        if nm == "Logistic regression":
            continue
        o = allf[allf.model == nm].sort_values("fold")
        for met in ["accuracy", "brier"]:
            dlt = o[met].values - ref[met].values
            t, p = corrected_t(dlt, n_tr, n_te)
            tests.append({"model": nm, "metric": met, "mean_diff_vs_LR": dlt.mean(), "t": t, "p": p})
    tests = pd.DataFrame(tests)
    for met in ["accuracy", "brier"]:   # Holm adjustment within metric
        m = tests.metric == met
        p = tests.loc[m, "p"].values
        order = np.argsort(p)
        adj = np.empty_like(p)
        run = 0
        for rank, i in enumerate(order):
            run = max(run, min(1, (len(p) - rank) * p[i]))
            adj[i] = run
        tests.loc[m, "p_holm"] = adj
    tests.to_csv(RES / "E2_tests_vs_LR.csv", index=False)
    print(tests.round(4).to_string(index=False))
    pd.DataFrame(rows).to_csv(RES / "E2_classifiers.csv", index=False)


# ---------------------------------------------------------------------------------------
def _nested_outer(i, tr, te):
    Xtr, ytr = X.iloc[tr], y.iloc[tr]
    inner = StratifiedKFold(5, shuffle=True, random_state=SEED + i)
    scores = {}
    for nm, clf in classifiers().items():
        p = make_pipeline(clf, F)
        s = []
        for itr, ite in inner.split(Xtr, ytr):
            m = clone(p).fit(Xtr.iloc[itr], ytr.iloc[itr])
            s.append(f1_score(ytr.iloc[ite], m.predict(Xtr.iloc[ite])))
        scores[nm] = np.mean(s)
    best = max(ORDER, key=lambda k: (round(scores[k], 10), -ORDER.index(k)))
    m = make_pipeline(classifiers()[best], F).fit(Xtr, ytr)
    pr = m.predict_proba(X.iloc[te])[:, 1]
    return {"fold": i, "selected": best, "inner_best_f1": scores[best],
            **metrics(y.iloc[te].values, pr), **{"inner_" + k: v for k, v in scores.items()}}


def E3_nested():
    """Nested CV of the complete model-selection procedure (outer 5x5, inner 5-fold)."""
    if done("E3_nested.csv"):
        return
    section("E3 nested cross-validation")
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    out = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(_nested_outer)(i, tr, te)
                                             for i, (tr, te) in enumerate(outer.split(X, y)))
    d = pd.DataFrame(out)
    d.to_csv(RES / "E3_nested.csv", index=False)
    print(d[["fold", "selected", "inner_best_f1", "f1", "accuracy", "errors"]].to_string(index=False))
    print("nested F1 mean", d.f1.mean(), "| mean inner best F1 (optimistic)", d.inner_best_f1.mean())
    print(d.selected.value_counts())


# ---------------------------------------------------------------------------------------
def E4_leakage():
    """Proper (all preprocessing in folds) vs two common leaky practices."""
    if done("E4_leakage.csv"):
        return
    section("E4 leakage experiments")
    models = {k: v for k, v in classifiers().items()
              if k in ["Logistic regression", "k-nearest neighbours", "Random forest"]}
    # Leak A: preprocessing + MI feature selection fitted on all 400 patients before CV
    pre = Pipeline(preprocessing(F, "mice"))
    Xa = pd.DataFrame(pre.fit_transform(X), columns=F)
    sel = SelectKBest(lambda a, b: mutual_info_classif(a, b, random_state=SEED), k=12).fit(Xa, y)
    Xa_sel = Xa.loc[:, sel.get_support()]
    # Leak B: class-conditional median imputation on all data (uses the label), then CV
    Xb = X.copy()
    for c in F:
        Xb[c] = Xb.groupby(y)[c].transform(lambda s: s.fillna(s.median()))
    rows, folds = [], []
    for nm, clf in models.items():
        variants = {
            "Proper (all steps inside folds)": (make_pipeline(clf, F, extra=[("select", SelectKBest(lambda a, b: mutual_info_classif(a, b, random_state=SEED), k=12))]), X),
            "Leak A: preprocessing and selection on all data": (Pipeline([("clf", clone(clf))]), Xa_sel),
            "Leak B: class-conditional imputation on all data": (Pipeline([("scale", StandardScaler()), ("clf", clone(clf))]), Xb),
            "Proper, label-free median imputation": (Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("clf", clone(clf))]), X),
        }
        for vn, (est, Xd) in variants.items():
            d = cv_eval(est, Xd)
            d["model"], d["variant"] = nm, vn
            folds.append(d)
            rows.append({"model": nm, "variant": vn, **summary(d)})
            print(f"{nm:22s} | {vn:50s} acc={d.accuracy.mean():.4f} f1={d.f1.mean():.4f} auc={d.auc.mean():.4f} err={d.errors.sum()}", flush=True)
    pd.DataFrame(rows).to_csv(RES / "E4_leakage.csv", index=False)
    pd.concat(folds).to_csv(RES / "E4_leakage_folds.csv", index=False)


# ---------------------------------------------------------------------------------------
def _split(seed, nm, est):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    m = clone(est).fit(Xtr, ytr)
    return {"seed": seed, "model": nm, **metrics(yte.values, m.predict_proba(Xte)[:, 1])}


def E5_split_variability(n_splits=200):
    """Test-set accuracy over many random stratified 80/20 splits."""
    if done("E5_splits.csv"):
        return
    section("E5 single-split variability")
    ests = {"Logistic regression": make_pipeline(LogisticRegression(max_iter=2000), F),
            "Extremely randomised trees": make_pipeline(classifiers()["Extremely randomised trees"], F),
            "Envelope rule": EnvelopeRule()}
    out = Parallel(n_jobs=N_JOBS)(delayed(_split)(s, nm, e) for nm, e in ests.items()
                                  for s in range(n_splits))
    d = pd.DataFrame(out)
    d.to_csv(RES / "E5_splits.csv", index=False)
    g = d.groupby("model").agg(acc_mean=("accuracy", "mean"), acc_min=("accuracy", "min"),
                               acc_p2_5=("accuracy", lambda s: s.quantile(.025)),
                               acc_p97_5=("accuracy", lambda s: s.quantile(.975)),
                               perfect=("errors", lambda s: (s == 0).mean()),
                               max_errors=("errors", "max"))
    print(g)


# ---------------------------------------------------------------------------------------
ABLATIONS = {
    "All 24 variables": FEATS,
    "Without hypertension and diabetes": [c for c in FEATS if c not in ["htn", "dm"]],
    "Without the 10 binary variables": [c for c in FEATS if c not in BIN],
    "Without anaemia-related variables (hemo, pcv, rc, ane)": [c for c in FEATS if c not in ["hemo", "pcv", "rc", "ane"]],
    "Without kidney markers (sc, bu, sg, al)": [c for c in FEATS if c not in ["sc", "bu", "sg", "al"]],
    "Urine dipstick only (sg, al, su)": ["sg", "al", "su"],
    "Serum creatinine only": ["sc"],
    "Haemoglobin only": ["hemo"],
    "Serum creatinine, haemoglobin, specific gravity": ["sc", "hemo", "sg"],
}


def E6_ablation():
    if done("E6_ablation.csv"):
        return
    section("E6 ablation")
    rows, folds = [], []
    for an, feats in ABLATIONS.items():
        feats = [c for c in FEATS if c in feats]
        Xs = X[feats]
        ests = {"Logistic regression": make_pipeline(LogisticRegression(max_iter=2000), feats),
                "Extremely randomised trees": make_pipeline(classifiers()["Extremely randomised trees"], feats),
                "Envelope rule": EnvelopeRule()}
        for nm, est in ests.items():
            d = cv_eval(est, Xs)
            d["model"], d["set"] = nm, an
            folds.append(d)
            rows.append({"set": an, "n_vars": len(feats), "model": nm, **summary(d)})
            print(f"{an:58s} {nm:28s} acc={d.accuracy.mean():.4f} auc={d.auc.mean():.4f}", flush=True)
    pd.DataFrame(rows).to_csv(RES / "E6_ablation.csv", index=False)
    pd.concat(folds).to_csv(RES / "E6_ablation_folds.csv", index=False)


# ---------------------------------------------------------------------------------------
def E7_imputation():
    if done("E7_imputation.csv"):
        return
    section("E7 imputation x indicators x downstream model")
    rows = []
    for imp in ["median", "knn", "mice"]:
        for ind in [False, True]:
            for nm in ["Logistic regression", "Extremely randomised trees"]:
                d = cv_eval(make_pipeline(classifiers()[nm], F, imputer=imp, add_indicator=ind))
                rows.append({"imputer": imp, "indicators": ind, "model": nm, **summary(d)})
                print(f"{imp:6s} ind={ind!s:5s} {nm:28s} acc={d.accuracy.mean():.4f} f1={d.f1.mean():.4f} "
                      f"auc={d.auc.mean():.4f} brier={d.brier.mean():.4f}", flush=True)
    pd.DataFrame(rows).to_csv(RES / "E7_imputation.csv", index=False)


# ---------------------------------------------------------------------------------------
def E8_permutation_test(n=100):
    if done("E8_permutation_test.json"):
        return
    section("E8 label-permutation test")
    p = make_pipeline(LogisticRegression(max_iter=2000), F)
    score, perm, pval = permutation_test_score(
        p, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="accuracy",
        n_permutations=n, random_state=SEED, n_jobs=N_JOBS)
    out = {"accuracy": score, "perm_mean": perm.mean(), "perm_sd": perm.std(ddof=1),
           "perm_max": perm.max(), "p": pval, "n_permutations": n,
           "majority_class_rate": float(y.mean())}
    json.dump(out, open(RES / "E8_permutation_test.json", "w"), indent=2, default=float)
    print(out)


# ---------------------------------------------------------------------------------------
def E9_holdout():
    """Development on the original 80/20 split (seed 42): select by CV on train, tune, test once."""
    if done("E9_holdout.json"):
        return
    section("E9 hold-out analysis (original protocol, corrected code)")
    from sklearn.model_selection import RandomizedSearchCV
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv3 = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    sc = {}
    for nm in ORDER:
        d = cv_eval(make_pipeline(classifiers()[nm], F), Xtr, ytr, cv3)
        sc[nm] = d.f1.mean()
    best = max(ORDER, key=lambda k: (round(sc[k], 10), -ORDER.index(k)))
    grid = {"clf__n_estimators": [300, 600], "clf__max_depth": [None, 6, 10], "clf__min_samples_leaf": [1, 2, 3]}
    final = make_pipeline(classifiers()[best], F)
    tuned = None
    if best == "Extremely randomised trees":
        s = RandomizedSearchCV(final, grid, n_iter=15, scoring="f1", cv=5, random_state=SEED,
                               n_jobs=N_JOBS).fit(Xtr, ytr)
        final, tuned = s.best_estimator_, {k: str(v) for k, v in s.best_params_.items()}
    else:
        final.fit(Xtr, ytr)
    p = final.predict_proba(Xte)[:, 1]
    m = metrics(yte.values, p)
    k = len(yte) - m["errors"]
    lo = stats.beta.ppf(0.025, k, len(yte) - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(0.975, k + 1, len(yte) - k) if k < len(yte) else 1.0
    pi = permutation_importance(final, Xte, yte, scoring="f1", n_repeats=30, random_state=SEED, n_jobs=N_JOBS)
    imp = pd.DataFrame({"var": F, "mean": pi.importances_mean, "sd": pi.importances_std}).sort_values("mean", ascending=False)
    imp.to_csv(RES / "E9_permutation_importance.csv", index=False)
    pre = Pipeline(preprocessing(F, "mice")).fit(Xtr)
    mi = pd.Series(mutual_info_classif(pre.transform(Xtr), ytr, random_state=SEED), index=F).sort_values(ascending=False)
    mi.to_csv(RES / "E9_mutual_information.csv", header=["mi"])
    env = EnvelopeRule().fit(Xtr, ytr)
    me = metrics(yte.values, env.predict_proba(Xte)[:, 1])
    out = {"cv_f1_train": sc, "selected": best, "tuned": tuned, "test": m,
           "test_acc_ci95": [lo, hi], "envelope_rule_test": me,
           "test_n": int(len(yte)), "test_ckd": int(yte.sum())}
    json.dump(out, open(RES / "E9_holdout.json", "w"), indent=2, default=float)
    print(json.dumps(out, indent=1, default=float))


# ---------------------------------------------------------------------------------------
def _kmedoids(Xm, k, iters=100, seed=SEED):
    rng = np.random.default_rng(seed)
    D = pairwise_distances(Xm)
    med = rng.choice(len(Xm), k, replace=False)
    for _ in range(iters):
        lab = D[:, med].argmin(1)
        new = np.array([np.where(lab == j)[0][D[np.ix_(lab == j, lab == j)].sum(1).argmin()] for j in range(k)])
        if set(new) == set(med):
            break
        med = new
    return D[:, med].argmin(1)


def E10_clustering():
    if done("E10_clustering.csv"):
        return
    section("E10 clustering (supplementary)")
    Xtr, _, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    Xall = Pipeline(preprocessing(F, "mice")).fit(Xtr).transform(X)
    yall = y.values
    sil = {k: silhouette_score(Xall, KMeans(k, n_init=20, random_state=SEED).fit_predict(Xall)) for k in range(2, 9)}
    nn = NearestNeighbors(n_neighbors=5).fit(Xall)
    eps = float(np.percentile(np.sort(nn.kneighbors(Xall)[0][:, -1]), 90))
    res = {"k-means (k=2)": KMeans(2, n_init=20, random_state=SEED).fit_predict(Xall),
           "k-medoids (k=2)": _kmedoids(Xall, 2),
           "Ward hierarchical (k=2)": AgglomerativeClustering(2, linkage="ward").fit_predict(Xall),
           f"DBSCAN (eps={eps:.1f})": DBSCAN(eps=eps, min_samples=5).fit_predict(Xall)}

    def purity(lab):
        return sum(np.bincount(yall[lab == c]).max() for c in np.unique(lab) if c != -1) / len(yall)
    rows = []
    for nm, lab in res.items():
        ok = len(set(lab) - {-1}) > 1
        rows.append({"method": nm, "clusters": len(set(lab) - {-1}), "noise": int((lab == -1).sum()),
                     "silhouette": silhouette_score(Xall[lab != -1], lab[lab != -1]) if ok else np.nan,
                     "ari": adjusted_rand_score(yall, lab), "purity": purity(lab)})
    d = pd.DataFrame(rows)
    d.to_csv(RES / "E10_clustering.csv", index=False)
    pd.Series(sil).to_csv(RES / "E10_silhouette_k.csv", header=["silhouette"])
    np.save(RES / "E10_labels.npy", np.vstack(list(res.values())))
    np.save(RES / "E10_pca2.npy", PCA(2, random_state=SEED).fit_transform(Xall))
    pca = PCA().fit(Pipeline(preprocessing(F, "mice")).fit_transform(Xtr))
    cum = np.cumsum(pca.explained_variance_ratio_)
    pd.Series(cum, index=range(1, len(cum) + 1)).to_csv(RES / "E10_pca_cumvar_train.csv", header=["cum"])
    print(d.round(3).to_string(index=False), "\nsilhouette by k", {k: round(v, 3) for k, v in sil.items()})


EXPERIMENTS = {"E0": E0_data_audit, "E1": E1_baselines, "E2": E2_classifiers, "E3": E3_nested,
               "E4": E4_leakage, "E5": E5_split_variability, "E6": E6_ablation,
               "E7": E7_imputation, "E8": E8_permutation_test, "E9": E9_holdout,
               "E10": E10_clustering}

if __name__ == "__main__":
    import platform
    import sklearn, scipy, joblib
    vers = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__}
    json.dump(vers, open(RES / "versions.json", "w"), indent=2)
    print(vers)
    todo = sys.argv[1:] or list(EXPERIMENTS)
    t0 = time.time()
    for e in todo:
        EXPERIMENTS[e]()
    print(f"\nFinished in {(time.time() - t0) / 60:.1f} min")
