"""RQ1: su->su, id->su, mixed (id train + k% su train) evaluated on su test. Plus id/su vocab overlap.
Features: word_tfidf. Hyperparameters are the fixed defaults in common.py
(no tuning, so test is touched once). Seeds 0-4 control su subsampling and model init.
Usage: python src/transfer.py
"""
import numpy as np
import pandas as pd

from common import MODELS, RESULTS, SEEDS, load, pipe, save, score, vectorizer

FEAT = "word_tfidf"
KS = [0, 10, 25, 50, 100]
id_tr, su_tr, su_te = load("id", "train"), load("su", "train"), load("su", "test")
out = {"feature": FEAT}


def run(tr_fn):
    res = {}
    for m in MODELS:
        f1s, accs = [], []
        for s in SEEDS:
            tr = tr_fn(s)
            r = score(su_te.label, pipe(FEAT, m, s).fit(tr.text, tr.label).predict(su_te.text))
            f1s.append(r["macro_f1"]); accs.append(r["acc"])
        res[m] = dict(macro_f1_mean=np.mean(f1s), macro_f1_std=np.std(f1s), acc_mean=np.mean(accs), acc_std=np.std(accs))
        print(m, f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}", flush=True)
    return res


for name, fn in [("su->su", lambda s: su_tr), ("id->su", lambda s: id_tr)] + [
    (f"mixed_k{k}", (lambda k: lambda s: pd.concat([id_tr, su_tr.sample(frac=k / 100, random_state=s)]))(k)) for k in KS
]:
    print(name); out[name] = run(fn)

# vocabulary overlap between id and su train sets
ov = {}
for kind in ["word", "char"]:
    v = {l: set(vectorizer(f"{kind}_bow").fit(d.text).vocabulary_) for l, d in [("id", id_tr), ("su", su_tr)]}
    a, b = v["id"], v["su"]
    ov[kind] = dict(n_id=len(a), n_su=len(b), n_shared=len(a & b), jaccard=len(a & b) / len(a | b), frac_su_in_id=len(a & b) / len(b))
    # token coverage: share of su test n-gram occurrences whose n-gram is in id train vocab
    X = vectorizer(f"{kind}_bow").fit(su_te.text)
    counts = np.asarray(X.transform(su_te.text).sum(0)).ravel()
    inv = X.get_feature_names_out()
    ov[kind]["su_test_token_coverage_by_id"] = float(counts[[g in a for g in inv]].sum() / counts.sum())
out["vocab_overlap"] = ov
save("transfer", out)

L = [f"Test: su test set, features {FEAT}, macro-F1 mean ± std over seeds {SEEDS}.\n",
     "| setting | " + " | ".join(MODELS) + " |", "|---|" + "---|" * len(MODELS)]
for k, v in out.items():
    if k not in ("feature", "vocab_overlap"):
        L.append(f"| {k} | " + " | ".join(f"{v[m]['macro_f1_mean']:.3f} ± {v[m]['macro_f1_std']:.3f}" for m in MODELS) + " |")
L += ["", "Vocabulary overlap (train sets; word = 1-2 grams, char = char_wb 2-5):\n",
      "| n-gram | #id | #su | shared | Jaccard | frac su in id | su-test occurrence coverage by id |", "|---|---|---|---|---|---|---|"]
for k, o in ov.items():
    L.append(f"| {k} | {o['n_id']} | {o['n_su']} | {o['n_shared']} | {o['jaccard']:.3f} | {o['frac_su_in_id']:.3f} | {o['su_test_token_coverage_by_id']:.3f} |")
(RESULTS / "transfer.md").write_text("\n".join(L) + "\n")
