"""RQ1 with character TF-IDF features:
su->su, id->su, mixed (id train + k% su train) evaluated on su test.
Features: char_tfidf (char_wb 2-5).
Hyperparameters: fixed defaults in common.py. Seeds 0-4 control su subsampling and model init.
Usage: python src/transfer_char.py
"""
import numpy as np
import pandas as pd

from common import MODELS, RESULTS, SEEDS, load, pipe, save, score

FEAT = "char_tfidf"
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
            f1s.append(r["macro_f1"])
            accs.append(r["acc"])
        res[m] = dict(
            macro_f1_mean=float(np.mean(f1s)),
            macro_f1_std=float(np.std(f1s)),
            acc_mean=float(np.mean(accs)),
            acc_std=float(np.std(accs)),
        )
        print(m, f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}", flush=True)
    return res


for name, fn in [("su->su", lambda s: su_tr), ("id->su", lambda s: id_tr)] + [
    (f"mixed_k{k}", (lambda k: lambda s: pd.concat([id_tr, su_tr.sample(frac=k / 100, random_state=s)]))(k)) for k in KS
]:
    print(name)
    out[name] = run(fn)

save("transfer_char", out)

L = [
    f"Test: su test set, features {FEAT}, macro-F1 mean ± std over seeds {SEEDS}.\n",
    "| setting | " + " | ".join(MODELS) + " |",
    "|---|" + "---|" * len(MODELS),
]
for k, v in out.items():
    if k not in ("feature", "vocab_overlap"):
        L.append(
            f"| {k} | "
            + " | ".join(f"{v[m]['macro_f1_mean']:.3f} ± {v[m]['macro_f1_std']:.3f}" for m in MODELS)
            + " |"
        )

(RESULTS / "transfer_char.md").write_text("\n".join(L) + "\n")
print("Saved to results/transfer_char.json and results/transfer_char.md")
