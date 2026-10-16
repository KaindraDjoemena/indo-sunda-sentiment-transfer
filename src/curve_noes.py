"""RQ3: MLP learning curve with early_stopping=True and early_stopping=False
(10/25/50/100% of su train -> su test).
Usage: python src/curve_noes.py -> results/curve_mlp_compare.{json,md}
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from common import RESULTS, SEEDS, load, save, score, vectorizer

FRACS = [0.10, 0.25, 0.50, 1.00]
tr, te = load("su", "train"), load("su", "test")
out = {}

for es in [True, False]:
    tag = "es" if es else "noes"
    for f in FRACS:
        f1s = []
        for s in SEEDS:
            sub = tr if f == 1.0 else tr.iloc[train_test_split(np.arange(len(tr)), train_size=f, stratify=tr.label, random_state=s)[0]]
            clf = MLPClassifier(hidden_layer_sizes=(64,), early_stopping=es, max_iter=300, random_state=s)
            p = Pipeline([("vec", vectorizer("word_tfidf")), ("clf", clf)])
            f1s.append(score(te.label, p.fit(sub.text, sub.label).predict(te.text))["macro_f1"])
        key = f"mlp_{tag}/{int(f*100)}"
        out[key] = {"early_stopping": es, "n_train": len(sub), "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)), "per_seed": [float(x) for x in f1s]}
        print(key, f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}", flush=True)

save("curve_mlp_compare", out)

lines = [
    "| Setting | 10% (n=50) | 25% (n=125) | 50% (n=250) | 100% (n=500) |",
    "|---|---|---|---|---|",
]
for es in [True, False]:
    tag = "es" if es else "noes"
    lbl = "MLP (early_stopping=True)" if es else "MLP (early_stopping=False)"
    row = [lbl]
    for f in FRACS:
        k = f"mlp_{tag}/{int(f*100)}"
        row.append(f"{out[k]['macro_f1_mean']:.3f} ± {out[k]['macro_f1_std']:.3f}")
    lines.append("| " + " | ".join(row) + " |")

(RESULTS / "curve_mlp_compare.md").write_text("MLP Learning Curve: Effect of early_stopping on su test Macro-F1\n\n" + "\n".join(lines) + "\n")
print("Saved to results/curve_mlp_compare.json and results/curve_mlp_compare.md")
