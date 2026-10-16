"""RQ2 feature ablation on Sundanese (su->su). Tune with stratified 5-fold CV on train+valid,
then fit on train+valid and evaluate test once per seed. Usage: python src/ablation.py [model ...]
(with model names, only those models are rerun and merged into the existing results/ablation.json;
 --resume keeps the configurations of those models that are already in the file; N_JOBS=k runs the CV folds in k processes)"""
import json
import os
import re
import sys
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from common import MODELS, RESULTS, SEEDS, load, pipe, save, score

FEATS = ["word_tfidf", "word_bow", "char_tfidf", "char_bow"]
NORMS = ["raw", "lower", "norm"]  # raw: no lowercasing; lower: lowercase; norm: lower + strip urls/@/punct/digits + squeeze repeats
GRIDS = {
    "svm": {"clf__C": [0.1, 1, 10]},
    "nb": {"clf__alpha": [0.01, 0.1, 1.0]},
    "rf": {"clf__min_samples_leaf": [1, 2]},
    "mlp": {"clf__alpha": [1e-4, 1e-2], "clf__hidden_layer_sizes": [(64,), (128,)]},
}


def normalise(t: str) -> str:
    t = re.sub(r"https?://\S+|@\w+", " ", t.lower())
    t = re.sub(r"[^\w\s]|\d", " ", t)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    return re.sub(r"\s+", " ", t).strip()


def prep(texts, norm):
    return texts.map(normalise) if norm == "norm" else texts


def build(feat, m, norm, seed=0):
    p = pipe(feat, m, seed)
    p.set_params(vec__lowercase=(norm != "raw"))
    return p


trv = pd.concat([load("su", "train"), load("su", "valid")], ignore_index=True)
te = load("su", "test")
cv = StratifiedKFold(5, shuffle=True, random_state=0)
RESUME = "--resume" in sys.argv
ARGS = [a for a in sys.argv[1:] if a != "--resume"]
RUN = ARGS or MODELS
old = json.loads((RESULTS / "ablation.json").read_text()) if ARGS else {}
out = {k: v for k, v in old.items() if RESUME or k.split("/")[0] not in RUN}
N_JOBS = int(os.environ.get("N_JOBS", "1"))
rows = []
for m, feat, norm in product(RUN, FEATS, NORMS):
    if f"{m}/{feat}/{norm}" in out:
        print("skip (done):", m, feat, norm, flush=True)
        continue
    X, Xt = prep(trv.text, norm), prep(te.text, norm)
    gs = GridSearchCV(build(feat, m, norm), GRIDS[m], scoring="f1_macro", cv=cv, n_jobs=N_JOBS, refit=False).fit(X, trv.label)
    best = gs.best_params_
    runs = [score(te.label, build(feat, m, norm, s).set_params(**best).fit(X, trv.label).predict(Xt)) for s in SEEDS]
    f1, acc = [r["macro_f1"] for r in runs], [r["acc"] for r in runs]
    out[f"{m}/{feat}/{norm}"] = dict(best_params={k: str(v) for k, v in best.items()}, cv_macro_f1=gs.best_score_,
        macro_f1_mean=np.mean(f1), macro_f1_std=np.std(f1), acc_mean=np.mean(acc), acc_std=np.std(acc))
    rows.append((m, feat, norm, gs.best_score_, np.mean(f1), np.std(f1), np.mean(acc), str(best)))
    print(rows[-1], flush=True)
    save("ablation", out)

out = {k: out[k] for k in sorted(out, key=lambda k: (MODELS.index(k.split("/")[0]), FEATS.index(k.split("/")[1]), NORMS.index(k.split("/")[2])))}
save("ablation", out)
md = ["| model | features | norm | CV macro-F1 | test macro-F1 (mean ± std) | test acc | best params |", "|---|---|---|---|---|---|---|"]
md += [f"| {k.split('/')[0]} | {k.split('/')[1]} | {k.split('/')[2]} | {v['cv_macro_f1']:.3f} | {v['macro_f1_mean']:.3f} ± {v['macro_f1_std']:.3f} | "
       f"{v['acc_mean']:.3f} | {v['best_params']} |" for k, v in out.items()]
(RESULTS / "ablation.md").write_text("\n".join(md) + "\n")
