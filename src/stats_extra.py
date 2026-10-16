"""Paired significance tests for the main RQ1/RQ2 claims (same design as stats.py: exact McNemar per seed,
paired bootstrap of the macro-F1 difference pooled over seeds). Fixed defaults from common.py; no tuning.
Comparisons:
  (a) char_tfidf vs word_tfidf under id->su, each model
  (b) char_tfidf vs word_tfidf under su->su, each model
  (c) mixed_k10 vs id->su (word_tfidf), each model  [su sample as in transfer.py]
  (d) mixed_k10 vs su-only 10% (word_tfidf), each model  [su sample as in curve.py]
  (e) RF id->su word: class_weight=None vs balanced
Holm correction over all comparisons, applied to the median McNemar p.
Usage: python src/stats_extra.py -> results/stats_extra.{json,md}
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from common import MODELS, RESULTS, SEEDS, load, pipe, save, vectorizer

B = 2000
id_tr, su_tr, te = load("id", "train"), load("su", "train"), load("su", "test")
y = te.label.to_numpy()


def su10(s):
    return su_tr.iloc[train_test_split(np.arange(len(su_tr)), train_size=0.10, stratify=su_tr.label, random_state=s)[0]]


TRAIN = {
    "id": lambda s: id_tr,
    "su": lambda s: su_tr,
    "mixed_k10": lambda s: pd.concat([id_tr, su_tr.sample(frac=0.10, random_state=s)]),
    "su10": su10,
}


def preds(setting, feat, m):
    out = []
    for s in SEEDS:
        tr = TRAIN[setting](s)
        if m == "rf_nocw":
            p = Pipeline([("vec", vectorizer(feat)), ("clf", RandomForestClassifier(n_estimators=300, class_weight=None, random_state=s, n_jobs=-1))])
        else:
            p = pipe(feat, m, s)
        out.append(p.fit(tr.text, tr.label).predict(te.text))
    return out


def compare(pa_all, pb_all):
    ps, ds, fa, fb, boots = [], [], [], [], []
    for s, pa, pb in zip(SEEDS, pa_all, pb_all):
        ca, cb = pa == y, pb == y
        n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        ps.append(binomtest(n01, n01 + n10, .5).pvalue if n01 + n10 else 1.0)
        f = lambda i: f1_score(y[i], pa[i], average="macro") - f1_score(y[i], pb[i], average="macro")
        rng = np.random.default_rng(s)
        boots += [f(rng.integers(0, len(y), len(y))) for _ in range(B)]
        ds.append(f(np.arange(len(y))))
        fa.append(f1_score(y, pa, average="macro")); fb.append(f1_score(y, pb, average="macro"))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"f1_a_mean": float(np.mean(fa)), "f1_b_mean": float(np.mean(fb)), "delta_f1_mean": float(np.mean(ds)),
            "ci95": [float(lo), float(hi)], "ci_excludes_0": bool(lo > 0 or hi < 0),
            "mcnemar_p_per_seed": ps, "mcnemar_p_median": float(np.median(ps)), "n_seeds_p<0.05": int(sum(p < .05 for p in ps))}


cache = {}


def P(*k):
    if k not in cache:
        cache[k] = preds(*k)
    return cache[k]


out = {}
for m in MODELS:
    out[f"(a) char vs word, id->su ({m})"] = compare(P("id", "char_tfidf", m), P("id", "word_tfidf", m))
for m in MODELS:
    out[f"(b) char vs word, su->su ({m})"] = compare(P("su", "char_tfidf", m), P("su", "word_tfidf", m))
for m in MODELS:
    out[f"(c) mixed_k10 vs id->su, word ({m})"] = compare(P("mixed_k10", "word_tfidf", m), P("id", "word_tfidf", m))
for m in MODELS:
    out[f"(d) mixed_k10 vs su-only 10%, word ({m})"] = compare(P("mixed_k10", "word_tfidf", m), P("su10", "word_tfidf", m))
out["(e) RF id->su word: class_weight None vs balanced"] = compare(P("id", "word_tfidf", "rf_nocw"), P("id", "word_tfidf", "rf"))

# Holm over median McNemar p-values
keys = sorted(out, key=lambda k: out[k]["mcnemar_p_median"])
n, running = len(keys), 0.0
for i, k in enumerate(keys):
    running = max(running, min(1.0, (n - i) * out[k]["mcnemar_p_median"]))
    out[k]["holm_p"] = running
# Holm over all paired tests reported in the paper (stats.py + this file), as one family
base = json.loads((RESULTS / "stats.json").read_text())
fam = {**{f"stats/{k}": v["mcnemar_p_median"] for k, v in base.items()}, **{k: v["mcnemar_p_median"] for k, v in out.items()}}
order, running, holm_all = sorted(fam, key=fam.get), 0.0, {}
for i, k in enumerate(order):
    running = max(running, min(1.0, (len(order) - i) * fam[k]))
    holm_all[k] = running
out["_holm_all"] = holm_all
save("stats_extra", out)

rows = ["| comparison (A vs B) | F1 A | F1 B | dF1 (A-B) | 95% CI | McNemar p (median) | Holm p | seeds p<.05 |", "|---|---|---|---|---|---|---|---|"]
for k, v in out.items():
    if k.startswith("_"):
        continue
    rows.append(f"| {k} | {v['f1_a_mean']:.3f} | {v['f1_b_mean']:.3f} | {v['delta_f1_mean']:+.3f} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | "
                f"{v['mcnemar_p_median']:.2g} | {v['holm_p']:.2g} | {v['n_seeds_p<0.05']}/{len(SEEDS)} |")
head = (f"Paired on 400 su test sentences; fixed defaults (common.py); bootstrap B={B} per seed, pooled over seeds 0-4; "
        f"exact McNemar per seed; Holm correction over the {n} median p-values.\n\n")
(RESULTS / "stats_extra.md").write_text(head + "\n".join(rows) + "\n")
print("\n".join(rows))
