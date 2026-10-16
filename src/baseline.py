"""Baseline (su->su) and RQ1 transfer (id->su), word TF-IDF.
Test is evaluated once here. Hyperparameter tuning belongs in ablation.py using CV on train+valid.
Usage: python src/baseline.py
"""
import numpy as np

from common import MODELS, SEEDS, load, pipe, save, score

out = {}
for setting, src, tgt in [("su->su", "su", "su"), ("id->su", "id", "su")]:
    tr = load(src, "train")
    te = load(tgt, "test")
    for m in MODELS:
        runs = [score(te.label, pipe("word_tfidf", m, s).fit(tr.text, tr.label).predict(te.text)) for s in SEEDS]
        f1 = [r["macro_f1"] for r in runs]
        out[f"{setting}/{m}"] = {"macro_f1_mean": np.mean(f1), "macro_f1_std": np.std(f1)}
        print(setting, m, f"{np.mean(f1):.3f} ± {np.std(f1):.3f}")
save("baseline", out)
