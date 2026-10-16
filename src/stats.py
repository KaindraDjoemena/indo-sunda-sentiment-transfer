"""RQ3 significance: exact McNemar + paired bootstrap CI of macro-F1 difference on the same 400 su test sentences.
Comparisons: (a) best-2 models by su->su baseline.json macro-F1, (b) id->su vs su->su for each model.
Per seed (stochastic models differ by seed); bootstrap resamples pooled over seeds.
Usage: python src/stats.py -> results/stats.{json,md}
"""
import json

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from common import MODELS, RESULTS, SEEDS, load, pipe, save

B = 2000
tr = {"su": load("su", "train"), "id": load("id", "train")}
te = load("su", "test")
y = te.label.to_numpy()
base = json.loads((RESULTS / "baseline.json").read_text())
best2 = sorted(MODELS, key=lambda m: -base[f"su->su/{m}"]["macro_f1_mean"])[:2]
print("best2:", best2)

P = {(src, m, s): pipe("word_tfidf", m, s).fit(tr[src].text, tr[src].label).predict(te.text) for src in tr for m in MODELS for s in SEEDS}


def compare(a, b):
    ps, ds, boots = [], [], []
    for s in SEEDS:
        pa, pb = P[(*a, s)], P[(*b, s)]
        ca, cb = pa == y, pb == y
        n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())  # a right/b wrong, a wrong/b right
        ps.append(binomtest(n01, n01 + n10, .5).pvalue if n01 + n10 else 1.0)
        f = lambda i: f1_score(y[i], pa[i], average="macro") - f1_score(y[i], pb[i], average="macro")
        rng = np.random.default_rng(s)
        boots += [f(rng.integers(0, len(y), len(y))) for _ in range(B)]
        ds.append(f(np.arange(len(y))))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"delta_f1_mean": np.mean(ds), "ci95": [lo, hi], "ci_excludes_0": bool(lo > 0 or hi < 0),
            "mcnemar_p_per_seed": ps, "mcnemar_p_median": float(np.median(ps)), "n_seeds_p<0.05": int(sum(p < .05 for p in ps))}


out = {f"{best2[0]} vs {best2[1]} (su->su)": compare(("su", best2[0]), ("su", best2[1]))}
for m in MODELS:
    out[f"su->su vs id->su ({m})"] = compare(("su", m), ("id", m))
save("stats", out)
rows = ["| comparison (A vs B) | dF1 (A-B) | 95% CI | McNemar p (median) | seeds p<.05 |", "|---|---|---|---|---|"]
for k, v in out.items():
    rows.append(f"| {k} | {v['delta_f1_mean']:+.3f} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | {v['mcnemar_p_median']:.2g} | {v['n_seeds_p<0.05']}/{len(SEEDS)} |")
(RESULTS / "stats.md").write_text(f"Paired on 400 su test sentences; bootstrap B={B} per seed, pooled over seeds 0-4; exact McNemar.\n\n" + "\n".join(rows) + "\n")
print("\n".join(rows))
