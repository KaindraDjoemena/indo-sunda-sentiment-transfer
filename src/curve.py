"""RQ3 learning curve: train on 10/25/50/100% of su train (stratified subsample per seed), test on su test.
Usage: python src/curve.py   -> results/curve.{json,md}, results/figs/curve.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from common import MODELS, RESULTS, SEEDS, load, pipe, save, score

FRACS = [0.10, 0.25, 0.50, 1.00]
tr, te = load("su", "train"), load("su", "test")
out = {}
for m in MODELS:
    for f in FRACS:
        f1s = []
        for s in SEEDS:
            sub = tr if f == 1.0 else tr.iloc[train_test_split(np.arange(len(tr)), train_size=f, stratify=tr.label, random_state=s)[0]]
            f1s.append(score(te.label, pipe("word_tfidf", m, s).fit(sub.text, sub.label).predict(te.text))["macro_f1"])
        out[f"{m}/{int(f*100)}"] = {"n_train": len(sub), "macro_f1_mean": np.mean(f1s), "macro_f1_std": np.std(f1s), "per_seed": f1s}
        print(m, f, f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}", flush=True)
save("curve", out)

lines = ["| model | " + " | ".join(f"{int(f*100)}% (n={out[f'{MODELS[0]}/{int(f*100)}']['n_train']})" for f in FRACS) + " |", "|---|" + "---|" * len(FRACS)]
for m in MODELS:
    lines.append(f"| {m} | " + " | ".join(f"{out[f'{m}/{int(f*100)}']['macro_f1_mean']:.3f} ± {out[f'{m}/{int(f*100)}']['macro_f1_std']:.3f}" for f in FRACS) + " |")
(RESULTS / "curve.md").write_text("Macro-F1 (mean ± std over seeds 0-4), su train subsample -> su test, word TF-IDF.\n\n" + "\n".join(lines) + "\n")

(RESULTS / "figs").mkdir(exist_ok=True)
for m in MODELS:
    mu = [out[f"{m}/{int(f*100)}"]["macro_f1_mean"] for f in FRACS]
    sd = [out[f"{m}/{int(f*100)}"]["macro_f1_std"] for f in FRACS]
    plt.errorbar([f * 100 for f in FRACS], mu, yerr=sd, marker="o", capsize=3, label=m)
plt.xscale("log"); plt.xticks([10, 25, 50, 100], ["10", "25", "50", "100"])
plt.xlabel("% of Sundanese training data"); plt.ylabel("Macro-F1 (su test)"); plt.legend(); plt.grid(alpha=.3)
plt.savefig(RESULTS / "figs" / "curve.png", dpi=200, bbox_inches="tight")
