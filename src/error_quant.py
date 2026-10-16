"""Quantitative error analysis for id->su with word_tfidf features (fixed hyperparameters, seeds 0-4):
error rate on the Sundanese test sentences, split by the number of Indonesian word features that a sentence
activates (at or below the median vs above it), for all sentences and for the Neutral sentences.
Usage: python src/error_quant.py -> results/error_quant.{json,md}
"""
import numpy as np

from common import MODELS, RESULTS, SEEDS, load, pipe, save, vectorizer

tr, te = load("id", "train"), load("su", "test")
y = te.label.to_numpy()
active = (vectorizer("word_tfidf").fit(tr.text).transform(te.text) > 0).sum(axis=1).A1
med = float(np.median(active))
groups = {"low": active <= med, "high": active > med}
neutral = y == "neutral"
out = {"active_features_median": med, "n_low": int(groups["low"].sum()), "n_high": int(groups["high"].sum()),
       "n_neutral_low": int((groups["low"] & neutral).sum()), "n_neutral_high": int((groups["high"] & neutral).sum())}
rows = []
for m in MODELS:
    err = {g: [] for g in ["all_low", "all_high", "neu_low", "neu_high"]}
    for s in SEEDS:
        wrong = pipe("word_tfidf", m, s).fit(tr.text, tr.label).predict(te.text) != y
        for g, mask in groups.items():
            err[f"all_{g}"].append(wrong[mask].mean())
            err[f"neu_{g}"].append(wrong[mask & neutral].mean())
    out[m] = {k: float(np.mean(v)) for k, v in err.items()}
    rows.append(f"| {m} | {out[m]['all_low']:.3f} | {out[m]['all_high']:.3f} | {out[m]['neu_low']:.3f} | {out[m]['neu_high']:.3f} |")
save("error_quant", out)
(RESULTS / "error_quant.md").write_text(
    f"id->su, word_tfidf, mean over seeds 0-4. Active Indonesian word features per su test sentence: median {med}; "
    f"low = at or below the median (n = {out['n_low']}, Neutral {out['n_neutral_low']}), "
    f"high = above (n = {out['n_high']}, Neutral {out['n_neutral_high']}).\n\n"
    "| model | error, all, low | error, all, high | error, Neutral, low | error, Neutral, high |\n|---|---|---|---|---|\n"
    + "\n".join(rows) + "\n")
print(open(RESULTS / "error_quant.md").read())
