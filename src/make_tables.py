"""Generate the LaTeX supplementary tables (S1 File) directly from ../results/.

Run after run_analysis.py:  python make_tables.py  -> ../results/S1_tables.tex
"""
import json

import pandas as pd

from ckd_common import NAMES, ROOT

RES = ROOT / "results"
f3 = lambda v: f"{v:.3f}"
out = []


def table(label, caption, header, rows, spec):
    out.append(r"\begin{table}[!ht]\centering\small")
    out.append(r"\caption{{\bf " + label + r".} " + caption + "}")
    out.append(r"\begin{tabular}{" + spec + r"}\toprule")
    out.append(" & ".join(header) + r"\\\midrule")
    for r in rows:
        out.append(" & ".join(str(x) for x in r) + r"\\")
    out.append(r"\bottomrule\end{tabular}\end{table}" + "\n")


# S1 pipeline
table("S1 Table", "Preprocessing pipeline. Every step was fitted on the training data of each fold.",
      ["Step", "Setting", "Learned"],
      [["Winsorising", r"$[Q_1-3\,\mathrm{IQR},\,Q_3+3\,\mathrm{IQR}]$; 11 continuous variables only (age, bp, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc)", "limits"],
       ["Log transform", r"$\log(1+x)$ of bgr, bu, sc, wc, pot, sod (negative values set to 0)", "none"],
       ["Imputation", r"MICE (IterativeImputer, Bayesian ridge, $\le$15 iterations, no posterior sampling, seed 42); or median; or $k$NN ($k=5$); optional missingness indicators", "models"],
       ["Standardisation", "zero mean, unit variance", "mean, s.d."],
       ["Selection (leakage experiment only)", "12 variables with highest mutual information (SelectKBest)", "selected set"]],
      r"p{3.2cm}p{9cm}p{1.8cm}")

# S2 settings
table("S2 Table", "Classifier settings (scikit-learn defaults otherwise; seed 42) and the hyperparameter grid of the hold-out protocol.",
      ["Classifier", "Settings"],
      [["Logistic regression", "L2 penalty, $C=1$, lbfgs, max. 2,000 iterations"],
       [r"$k$-nearest neighbours", "$k=7$, uniform weights, Euclidean distance"],
       ["Support vector machine", "RBF kernel, $C=3$, gamma = scale, Platt probabilities"],
       ["Random forest", "500 trees"],
       ["Extremely randomised trees", "500 trees; hold-out grid: trees 300/600, max. depth none/6/10, min. leaf 1/2/3 (15 of 18 combinations sampled)"],
       ["Gradient boosting", "histogram-based, default settings"],
       ["Stacked ensemble", "SVM + random forest + gradient boosting; logistic-regression combiner; 5-fold internal CV"],
       ["Envelope rule", "min/max of each variable among training controls; CKD if any observed value outside"]],
      r"p{3.6cm}p{10.4cm}")

# S3 tests
t = pd.read_csv(RES / "E2_tests_vs_LR.csv")
rows = [[r.model, r.metric, f"{r.mean_diff_vs_LR:+.4f}", f"{r.t:.2f}", f"{r.p:.4f}", f"{r.p_holm:.4f}"] for r in t.itertuples()]
table("S3 Table", "Corrected resampled $t$-tests (Nadeau--Bengio; 50 folds, $n_\\mathrm{test}/n_\\mathrm{train}=0.25$) of each model against logistic regression; Holm adjustment within metric. The Brier score of the envelope rule is not a calibrated probability.",
      ["Model", "Metric", "Mean difference", "$t$", "$p$", "$p$ (Holm)"], rows, "llrrrr")

# S4 nested
n = pd.read_csv(RES / "E3_nested.csv")
rows = [[r.fold + 1, r.selected, f3(r.inner_best_f1), f3(r.accuracy), f3(r.f1), r.errors] for r in n.itertuples()]
rows.append([r"\midrule Mean", "", f3(n.inner_best_f1.mean()), f3(n.accuracy.mean()), f3(n.f1.mean()), n.errors.sum()])
table("S4 Table", "Nested cross-validation (outer stratified 5-fold repeated 5 times; inner stratified 5-fold over seven classifiers; ties broken in favour of the simpler model). Each outer test fold has 80 patients.",
      ["Outer fold", "Selected", "Inner F1", "Outer accuracy", "Outer F1", "Errors"], rows, "rlrrrr")

# S5 hold-out
h = json.load(open(RES / "E9_holdout.json"))
rows = [[k, f3(v)] for k, v in h["cv_f1_train"].items()]
rows += [[r"\midrule Selected model", h["selected"]],
         ["Tuned settings", ", ".join(f"{k.replace('clf__', '').replace('_', ' ')}={v}" for k, v in h["tuned"].items())],
         ["Test accuracy (80 patients)", f"{h['test']['accuracy']:.3f} (95\\% CI {h['test_acc_ci95'][0]*100:.1f}--{h['test_acc_ci95'][1]*100:.0f}\\%)"],
         ["Test AUC / Brier", f"{h['test']['auc']:.3f} / {h['test']['brier']:.4f}"],
         ["Envelope rule, test accuracy", f"{h['envelope_rule_test']['accuracy']:.3f} ({h['envelope_rule_test']['errors']} errors)"]]
table("S5 Table", "Conventional hold-out protocol (stratified 80/20 split, seed 42). Upper part: cross-validated F1 on the 320 training patients (5-fold, 3 repeats).",
      ["Item", "Value"], rows, "ll")

# S6 leakage
e = pd.read_csv(RES / "E4_leakage.csv")
rows = [[r.model, r.variant, f"{r.accuracy_mean:.4f} ({r.accuracy_sd:.4f})", f"{r.f1_mean:.4f}", f"{r.auc_mean:.4f}", r.errors_total] for r in e.itertuples()]
table("S6 Table", "Leakage experiments (same 50 folds; 4,000 predictions). Leak A: MICE pipeline and mutual-information selection fitted on all 400 patients before cross-validation. Leak B: class-conditional median imputation on all data.",
      ["Model", "Variant", "Accuracy (s.d.)", "F1", "AUC", "Errors"], rows, r"lp{5.6cm}rrrr")

# S7 ablation
a = pd.read_csv(RES / "E6_ablation.csv")
rows = [[r.set, r.n_vars, r.model, f3(r.accuracy_mean), f3(r.recall_mean), f3(r.specificity_mean), f3(r.auc_mean), r.errors_total] for r in a.itertuples()]
table("S7 Table", "Ablation (5-fold cross-validation repeated ten times).",
      ["Variable set", "$n$", "Model", "Acc.", "Sens.", "Spec.", "AUC", "Errors"], rows, r"p{4.6cm}rlrrrrr")

# S8 imputation
m = pd.read_csv(RES / "E7_imputation.csv")
rows = [[r.imputer.upper() if r.imputer != "knn" else "$k$NN", "yes" if r.indicators else "no", r.model, f"{r.accuracy_mean:.4f}", f"{r.f1_mean:.4f} ({r.f1_sd:.4f})", f"{r.brier_mean:.4f}", r.errors_total] for r in m.itertuples()]
table("S8 Table", "Imputation methods with and without missingness indicators (5-fold cross-validation repeated ten times). All AUCs were 1.000.",
      ["Imputer", "Indicators", "Model", "Accuracy", "F1 (s.d.)", "Brier", "Errors"], rows, "lllrrrr")

# S9 clustering
c = pd.read_csv(RES / "E10_clustering.csv")
rows = [[r.method, r.clusters, r.noise, "---" if pd.isna(r.silhouette) else f3(r.silhouette), f3(r.ari), f3(r.purity)] for r in c.itertuples()]
table("S9 Table", "Clustering of all 400 patients after preprocessing fitted on the training set of the hold-out split (labels not used for fitting). ARI, adjusted Rand index.",
      ["Method", "Clusters", "Noise", "Silhouette", "ARI", "Purity"], rows, "lrrrrr")

# S10 outliers
o = json.load(open(RES / "E0_audit.json"))["iqr15_outliers"]
rows = [[NAMES[k], v] for k, v in sorted(o.items(), key=lambda kv: -kv[1])]
table("S10 Table", "Number of values outside $Q_1-1.5\\,\\mathrm{IQR}$ and $Q_3+1.5\\,\\mathrm{IQR}$ for the 11 continuous variables (all 400 records, after the plausibility rule). These were winsorised at 3 IQR inside the pipeline, not removed.",
      ["Variable", "Values"], rows, "lr")

(RES / "S1_tables.tex").write_text("\n".join(out))
print("written", RES / "S1_tables.tex")
