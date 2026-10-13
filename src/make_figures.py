"""Figures for the manuscript, built only from the files in ../results/ (plus the data for Fig S1).

Run after run_analysis.py:  python make_figures.py
Writes PNG (preview) and 300-dpi TIFF (PLOS ONE submission) files to ../figures/.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from PIL import Image

from ckd_common import BIN, CONT, FEATS, NAMES, ROOT, load_data

RES = ROOT / "results"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
                     "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "figure.dpi": 150})
C_CKD, C_CTRL, C_ML, C_BASE, C_GREY = "#C8553D", "#2E6F95", "#3D5A80", "#E09F3E", "#8D99AE"
W1, W2 = 5.2, 7.5          # PLOS ONE widths (inches): ~1 column-and-a-half, full page


def panel(ax, letter, x=-0.12, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.tif", bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    im = Image.open(OUT / f"{name}.tif").convert("RGB")       # PLOS: RGB, LZW, no alpha
    if im.size[0] > 2250:                                      # PLOS max width 7.5 in at 300 dpi
        im = im.resize((2250, round(im.size[1] * 2250 / im.size[0])), Image.LANCZOS)
    im.save(OUT / f"{name}.tif", compression="tiff_lzw", dpi=(300, 300))
    plt.close(fig)


# ---------------------------------------------------------------------------------- Fig 1
def fig1():
    fig, ax = plt.subplots(figsize=(W2, 3.8))
    ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")

    def box(x, y, w, h, text, fc="#EEF2F7", ec=C_ML, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=5.6,
                fontweight="bold" if bold else "normal", wrap=True)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=1, color="#333"))

    box(1, 30, 20, 11, "UCI CKD data\n400 patients, 24 variables\n250 CKD / 150 not CKD", bold=True)
    box(26, 30, 22, 11, "Rule-based cleaning\n(no learned parameters)\nwhitespace, types, encoding,\nimplausible values → missing")
    box(53, 30, 21, 11, "Data audit\nmissingness by class,\ncontrol-group ranges")
    box(78, 30, 21, 11, "Reference-envelope rule\nand simple baselines\n(trees, missingness only)", fc="#FBF0DF", ec=C_BASE)
    arrow(21.5, 35.5, 25.5, 35.5); arrow(48.5, 35.5, 52.5, 35.5); arrow(74.5, 35.5, 77.5, 35.5)
    box(1, 2, 30, 22, "Leakage-free pipeline\n(fitted inside every training fold)\n\n"
        "winsorise 11 continuous variables (3 IQR)\n→ log(1+x) of 6 skewed variables\n"
        "→ imputation (median / kNN / MICE,\n     ± missingness indicators)\n→ standardisation → classifier", fc="#E8F1EA", ec="#4C956C")
    box(36, 13, 29, 11, "Repeated 5-fold CV (×10, 50 folds)\n7 classifiers, envelope rule;\nimputation and ablation experiments")
    box(36, 2, 29, 9, "Nested CV of model selection\n(outer 5-fold ×5, inner 5-fold)")
    box(70, 13, 29, 11, "Leakage experiments\n(preprocessing/selection on all data;\nclass-conditional imputation)", fc="#F8E1DC", ec=C_CKD)
    box(70, 2, 29, 9, "200 random 80/20 splits;\nlabel-permutation test;\noriginal hold-out protocol")
    arrow(37, 29.5, 20, 24.5); arrow(37, 29.5, 50, 24.5)
    arrow(31.5, 13, 35.5, 18); arrow(31.5, 8, 35.5, 7); arrow(65.5, 18, 69.5, 18); arrow(65.5, 7, 69.5, 7)
    save(fig, "Fig1")


# ---------------------------------------------------------------------------------- Fig 2
def fig2():
    v = pd.read_csv(RES / "E0_variables.csv").set_index("var")
    X, y = load_data()
    fig = plt.figure(figsize=(W2, 6.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], height_ratios=[1.7, 1], wspace=0.55, hspace=0.45)
    # A: missingness by class
    ax = fig.add_subplot(gs[:, 0])
    o = v.sort_values("missing_pct", ascending=True)
    yy = np.arange(len(o))
    ax.barh(yy + 0.2, o.missing_pct_ckd, 0.4, color=C_CKD, label="CKD (n=250)")
    ax.barh(yy - 0.2, o.missing_pct_ctrl, 0.4, color=C_CTRL, label="Not CKD (n=150)")
    ax.set_yticks(yy, [NAMES[i] for i in o.index]); ax.set_xlabel("Missing values (%)")
    ax.legend(frameon=False, loc="lower right"); panel(ax, "A", -0.62)
    # B: share of CKD patients outside the control range
    ax = fig.add_subplot(gs[0, 1])
    o = v.sort_values("ckd_outside_ctrl_range_pct_observed", ascending=True)
    col = [C_BASE if i in BIN else C_ML for i in o.index]
    ax.barh(np.arange(len(o)), o.ckd_outside_ctrl_range_pct_observed, color=col)
    ax.set_yticks(np.arange(len(o)), [NAMES[i] for i in o.index], fontsize=6)
    ax.set_xlabel("CKD patients with a value outside\nthe range seen in controls (%)")
    ax.set_xlim(0, 100)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_ML, label="continuous / ordinal"), Patch(color=C_BASE, label="binary (abnormal/yes)")],
              frameon=False, loc="lower right", fontsize=6)
    panel(ax, "B", -0.75)
    # C: number of excursions per patient
    ax = fig.add_subplot(gs[1, 1])
    from ckd_common import EnvelopeRule
    n = (EnvelopeRule().fit(X, y)._excursions(X) > 0).sum(1)
    bins = np.arange(-0.5, n.max() + 1.5)
    ax.hist(n[y.values == 1], bins=bins, color=C_CKD, label="CKD")
    ax.hist(n[y.values == 0], bins=bins, color=C_CTRL, label="Not CKD")
    ax.set_xlabel("Variables outside the control range\n(per patient)"); ax.set_ylabel("Patients")
    ax.legend(frameon=False); panel(ax, "C", -0.3)
    save(fig, "Fig2")


# ---------------------------------------------------------------------------------- Fig 3
def fig3():
    e2 = pd.read_csv(RES / "E2_classifiers.csv")
    e1 = pd.read_csv(RES / "E1_baselines.csv")
    d = pd.concat([e2.assign(kind="ml"), e1.assign(kind="base")]).sort_values("accuracy_mean")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.4), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, met, lab, letter in [(axes[0], "accuracy", "Accuracy", "A"), (axes[1], "auc", "ROC AUC", "B")]:
        yy = np.arange(len(d))
        c = [C_ML if k == "ml" else C_BASE for k in d.kind]
        ax.errorbar(d[f"{met}_mean"], yy, xerr=d[f"{met}_sd"], fmt="none", ecolor="#555", lw=0.8, capsize=2)
        ax.scatter(d[f"{met}_mean"], yy, c=c, s=22, zorder=3)
        for i, (m, s) in enumerate(zip(d[f"{met}_mean"], d[f"{met}_sd"])):
            ax.text(1.008, i, f"{m:.3f}", va="center", fontsize=6)
        ax.set_xlim(0.74, 1.035); ax.set_xticks([0.75, 0.8, 0.85, 0.9, 0.95, 1.0]); ax.set_xlabel(f"{lab} (mean ± s.d., 50 folds)")
        ax.axvline(1, color="#bbb", lw=0.6, ls=":")
        panel(ax, letter, -0.05 if letter == "B" else -0.75)
    axes[0].set_yticks(np.arange(len(d)), d.model)
    from matplotlib.lines import Line2D
    axes[1].legend(handles=[Line2D([], [], marker="o", ls="", color=C_ML, label="Machine-learning pipeline (MICE)"),
                            Line2D([], [], marker="o", ls="", color=C_BASE, label="Rule / simple baseline")],
                   frameon=False, loc="upper center", bbox_to_anchor=(-0.05, -0.2), ncol=2)
    save(fig, "Fig3")


# ---------------------------------------------------------------------------------- Fig 4
def fig4():
    e4 = pd.read_csv(RES / "E4_leakage.csv")
    e5 = pd.read_csv(RES / "E5_splits.csv")
    e3 = pd.read_csv(RES / "E3_nested.csv")
    fig = plt.figure(figsize=(W2, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 0.75], wspace=0.6)
    ax = fig.add_subplot(gs[0])
    variants = ["Proper (all steps inside folds)", "Leak A: preprocessing and selection on all data",
                "Proper, label-free median imputation", "Leak B: class-conditional imputation on all data"]
    short = ["Proper\n(MICE + MI-12)", "Leak A", "Proper\n(median)", "Leak B"]
    cols = ["#4C956C", C_CKD, "#4C956C", C_CKD]
    models = ["Logistic regression", "k-nearest neighbours", "Random forest"]
    w = 0.2
    for j, (vn, cl) in enumerate(zip(variants, cols)):
        vals = [e4[(e4.model == m) & (e4.variant == vn)].errors_total.iloc[0] for m in models]
        bars = ax.bar(np.arange(3) + (j - 1.5) * w, vals, w, color=cl, alpha=1 if j < 2 else 0.55,
                      hatch=None if j < 2 else "///", edgecolor="white", lw=0.3)
        for xi, vv in zip(np.arange(3) + (j - 1.5) * w, vals):
            ax.text(xi, vv + 1.5, str(int(vv)), ha="center", fontsize=5.5)
    ax.set_xticks(np.arange(3), ["Logistic\nregression", "k-nearest\nneighbours", "Random\nforest"])
    ax.set_ylabel("Misclassifications\n(4000 predictions, 50 folds)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4C956C", label="Proper (MICE pipeline + MI selection)"),
                       Patch(color=C_CKD, label="Leak A: fitted on all data"),
                       Patch(facecolor="#4C956C", alpha=.55, hatch="///", label="Proper (median imputation)"),
                       Patch(facecolor=C_CKD, alpha=.55, hatch="///", label="Leak B: class-wise imputation")],
              frameon=False, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    panel(ax, "A", -0.3)
    ax = fig.add_subplot(gs[1])
    for nm, cl in [("Logistic regression", C_ML), ("Extremely randomised trees", "#98C1D9"), ("Envelope rule", C_BASE)]:
        s = e5[e5.model == nm].errors
        cnt = s.value_counts().reindex(range(0, int(e5.errors.max()) + 1), fill_value=0)
        ax.plot(cnt.index, cnt.values / len(s) * 100, "o-", ms=3, lw=1, color=cl,
                label=nm.replace("Extremely randomised trees", "Extra trees"))
    ax.set_xlabel("Errors among 80 test patients"); ax.set_ylabel("Random splits (%)")
    ax.legend(frameon=False, fontsize=6, loc="upper right"); panel(ax, "B", -0.3)
    ax = fig.add_subplot(gs[2])
    vc = e3.selected.value_counts()
    ax.barh(np.arange(len(vc)), vc.values, color=C_ML)
    ax.set_yticks(np.arange(len(vc)), [s.replace("Extremely randomised trees", "Extra trees") for s in vc.index])
    ax.set_xlabel("Times selected\n(25 outer folds)")
    panel(ax, "C", -0.9)
    save(fig, "Fig4")


# ---------------------------------------------------------------------------------- Fig 5
def fig5():
    e6 = pd.read_csv(RES / "E6_ablation.csv")
    sets = list(dict.fromkeys(e6.set))
    models = ["Logistic regression", "Extremely randomised trees", "Envelope rule"]
    M = np.array([[e6[(e6.set == s) & (e6.model == m)].accuracy_mean.iloc[0] for m in models] for s in sets])
    fig, ax = plt.subplots(figsize=(W1, 3.4))
    im = ax.imshow(M, cmap="viridis", vmin=0.80, vmax=1.0, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=6.5,
                    color="white" if M[i, j] < 0.93 else "black")
    ax.set_xticks(range(3), ["Logistic\nregression", "Extra\ntrees", "Envelope\nrule"])
    ax.set_yticks(range(len(sets)), sets, fontsize=6.5)
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.set_label("Accuracy (repeated CV)")
    save(fig, "Fig5")


# ---------------------------------------------------------------------------------- S figs
def figS1_importance():
    mi = pd.read_csv(RES / "E9_mutual_information.csv", index_col=0)["mi"].head(12)[::-1]
    pi = pd.read_csv(RES / "E9_permutation_importance.csv").head(12)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.0), gridspec_kw={"wspace": 0.75})
    axes[0].barh([NAMES[i] for i in mi.index], mi.values, color="#4C956C")
    axes[0].set_xlabel("Mutual information with the label")
    axes[1].barh([NAMES[i] for i in pi["var"]], pi["mean"], xerr=pi["sd"], color=C_BASE, ecolor="#333", capsize=1.5)
    axes[1].set_xlabel("Drop in test F1 when permuted")
    panel(axes[0], "A", -0.6); panel(axes[1], "B", -0.6)
    save(fig, "S1_Fig")


def figS2_clustering():
    d = pd.read_csv(RES / "E10_clustering.csv")
    sil = pd.read_csv(RES / "E10_silhouette_k.csv", index_col=0)["silhouette"]
    P = np.load(RES / "E10_pca2.npy"); L = np.load(RES / "E10_labels.npy")
    _, y = load_data()
    fig, axes = plt.subplots(2, 3, figsize=(W2, 4.6), gridspec_kw={"hspace": 0.6, "wspace": 0.4})
    ax = axes[0, 0]; ax.plot(sil.index, sil.values, "o-", color=C_ML, ms=3)
    ax.set_xticks(sil.index); ax.set_xlabel("Number of clusters k"); ax.set_ylabel("Mean silhouette"); panel(ax, "A", -0.3)
    ax = axes[0, 1]
    for lab, c, n in [(1, C_CKD, "CKD"), (0, C_CTRL, "Not CKD")]:
        m = y.values == lab; ax.scatter(P[m, 0], P[m, 1], s=4, c=c, label=n, alpha=.7, lw=0)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.legend(frameon=False, loc="upper left", markerscale=2)
    panel(ax, "B", -0.3)
    for k, (ax, (_, r)) in enumerate(zip([axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]], d.iterrows())):
        lab = L[k]
        ax.scatter(P[:, 0], P[:, 1], s=4, c=np.where(lab == -1, "black", np.where(lab == 0, C_GREY, C_BASE)), lw=0)
        ax.set_title(f"{r['method']}\nARI {r['ari']:.2f}; purity {r['purity']:.2f}", fontsize=7)
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); panel(ax, "CDEF"[k], -0.3, 1.2)
    save(fig, "S2_Fig")


def figS3_pca():
    c = pd.read_csv(RES / "E10_pca_cumvar_train.csv", index_col=0)["cum"] * 100
    n95 = int((c >= 95).idxmax())
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    ax.plot(c.index, c.values, "o-", ms=3, color=C_ML)
    ax.axhline(95, ls="--", lw=0.8, color=C_CKD); ax.axvline(n95, ls="--", lw=0.8, color=C_CKD)
    ax.set_xlabel("Number of principal components"); ax.set_ylabel("Cumulative variance (%)")
    ax.text(n95 + 0.5, 60, f"{n95} components\nfor 95%", fontsize=7)
    save(fig, "S3_Fig")


if __name__ == "__main__":
    for f in [fig1, fig2, fig3, fig4, fig5, figS1_importance, figS2_clustering, figS3_pca]:
        f(); print("done", f.__name__)
