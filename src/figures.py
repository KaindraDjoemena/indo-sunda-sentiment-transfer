"""Paper figures from results/ files only (no training).
- results/figs/method.png: experiment flow (Fig. 1)
- results/figs/confusion_id_su.png: id->su confusion matrices, word features, seed 0 (Fig. 2)
- results/figs/curve_paper.png: su->su learning curve incl. MLP without early stopping (Fig. 3)
Usage: python src/figures.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

from common import RESULTS

FIGS = RESULTS / "figs"
FIGS.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "font.size": 9})
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"

# ---------- Fig. 1: experiment flow ----------
fig, ax = plt.subplots(figsize=(6.6, 3.5))
ax.set_xlim(0, 100); ax.set_ylim(-2, 61); ax.axis("off")


def box(x, y, w, h, title, lines):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc="white", ec=INK, lw=0.9))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=8.5, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 6.6, "\n".join(lines), ha="center", va="top", fontsize=7.5, color=INK, linespacing=1.35)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.9, color=INK))


box(1, 42, 30, 17, "NusaX-senti data", ["id and su (parallel)", "500 train / 100 valid", "400 test per language"])
box(35, 42, 30, 17, "Preprocessing", ["raw, lower, norm", "(norm: URLs, mentions,", "punctuation, digits)"])
box(69, 42, 30, 17, "Feature representation", ["word 1-2 grams", "character char_wb 2-5", "TF-IDF or BoW"])
box(1, 0, 30, 22, "RQ1: Transfer", ["su->su, id->su, mixed_k", "k = 10, 25, 50, 100%", "word and character features", "fixed hyperparameters"])
box(35, 0, 30, 22, "RQ2: Feature ablation", ["4 models x 4 features x", "3 preprocessing levels, su->su", "5-fold CV on", "train+valid (600)"])
box(69, 0, 30, 22, "RQ3: Learning curve", ["10/25/50/100% of su data", "McNemar test and", "paired bootstrap", "fixed hyperparameters"])
arrow(31.5, 51, 34.5, 51); arrow(65.5, 51, 68.5, 51)
ax.add_patch(FancyBboxPatch((1, 28), 98, 8, boxstyle="round,pad=0.4,rounding_size=1.2", fc="#f2f1ec", ec=INK, lw=0.9))
ax.text(50, 32, "SVM, Naive Bayes, Random Forest, MLP; 5 seeds (0-4); macro-F1 and accuracy on 400 su test sentences",
        ha="center", va="center", fontsize=7.8, color=INK)
arrow(84, 41.5, 84, 36.6)
for xc in (16, 50, 84):
    arrow(xc, 27.4, xc, 22.6)
fig.savefig(FIGS / "method.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------- Fig. 3: learning curve ----------
curve = json.loads((RESULTS / "curve.json").read_text())
mlp = json.loads((RESULTS / "curve_mlp_compare.json").read_text())
FR = [10, 25, 50, 100]
N = [50, 125, 250, 500]
series = [  # (label, means, stds, color, marker, linestyle)
    ("SVM", [curve[f"svm/{f}"]["macro_f1_mean"] for f in FR], [curve[f"svm/{f}"]["macro_f1_std"] for f in FR], "#2a78d6", "o", "-"),
    ("Naive Bayes", [curve[f"nb/{f}"]["macro_f1_mean"] for f in FR], [curve[f"nb/{f}"]["macro_f1_std"] for f in FR], "#eb6834", "s", "-"),
    ("Random Forest", [curve[f"rf/{f}"]["macro_f1_mean"] for f in FR], [curve[f"rf/{f}"]["macro_f1_std"] for f in FR], "#1baf7a", "^", "-"),
    ("MLP without early stopping", [mlp[f"mlp_noes/{f}"]["macro_f1_mean"] for f in FR], [mlp[f"mlp_noes/{f}"]["macro_f1_std"] for f in FR], "#eda100", "D", "-"),
    ("MLP with early stopping", [mlp[f"mlp_es/{f}"]["macro_f1_mean"] for f in FR], [mlp[f"mlp_es/{f}"]["macro_f1_std"] for f in FR], "#e87ba4", "v", "--"),
]
fig, ax = plt.subplots(figsize=(6.0, 3.4))
for lab, mu, sd, c, mk, ls in series:
    ax.errorbar(N, mu, yerr=sd, color=c, marker=mk, ms=5, lw=2, ls=ls, capsize=2.5, elinewidth=0.8,
                mec="white", mew=0.8, label=lab)
ax.set_xscale("log")
ax.set_xticks(N); ax.minorticks_off(); ax.set_xticklabels([f"{n}\n({f}%)" for n, f in zip(N, FR)])
ax.set_xlim(42, 600)
ax.set_ylim(0.1, 0.8)
ax.set_xlabel("Number of Sundanese training sentences (log scale)", color=INK)
ax.set_ylabel("Macro-F1 on su test data", color=INK)
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(MUTED)
ax.tick_params(colors=MUTED, labelcolor=INK)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=3, fontsize=7.5, frameon=False)
fig.savefig(FIGS / "curve_paper.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------- Fig. 2: id->su confusion matrices (word features, seed 0) ----------
err = json.loads((RESULTS / "error_analysis.json").read_text())
MODELS = [("svm", "SVM"), ("nb", "Naive Bayes"), ("rf", "Random Forest"), ("mlp", "MLP")]
CLASSES = ["Neg", "Neu", "Pos"]
# light-to-dark blue for the share of the true class
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
fig, axes = plt.subplots(1, 4, figsize=(6.6, 2.05), gridspec_kw={"wspace": 0.12})
for ax, (key, name) in zip(axes, MODELS):
    cm = err[key]["confusion_matrix"]
    share = [[v / sum(row) for v in row] for row in cm]  # row-normalised: share of the true class
    im = ax.imshow(share, cmap=BLUES, vmin=0, vmax=1)
    for r in range(3):
        for c in range(3):
            ax.text(c, r, str(cm[r][c]), ha="center", va="center", fontsize=8,
                    color="white" if share[r][c] > 0.55 else INK)
    ax.set_title(f"{name}\nmacro-F1 {err[key]['macro_f1']:.3f}", fontsize=8, color=INK, pad=4)
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES, fontsize=7.5)
    ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES if key == "svm" else [], fontsize=7.5)
    ax.set_xlabel("Predicted", fontsize=7.5, color=INK, labelpad=2)
    ax.tick_params(length=0, colors=INK)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, 3)], minor=True); ax.set_yticks([y - 0.5 for y in range(1, 3)], minor=True)
    ax.grid(which="minor", color="white", lw=1.5); ax.tick_params(which="minor", length=0)
axes[0].set_ylabel("True", fontsize=7.5, color=INK, labelpad=2)
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, ticks=[0, 0.5, 1])
cb.ax.tick_params(labelsize=7, colors=INK, length=2); cb.outline.set_visible(False)
cb.set_label("Share of true class", fontsize=7.5, color=INK)
fig.savefig(FIGS / "confusion_id_su.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved results/figs/method.png, results/figs/confusion_id_su.png and results/figs/curve_paper.png")
