"""Choose the MLP early_stopping setting without the test set: stratified 5-fold CV macro-F1 on train+valid,
seeds 0-4, word_tfidf and char_tfidf, for Sundanese (the RQ2 data) and, as a check, Indonesian (the id->su source).
The setting with the higher mean CV macro-F1 on Sundanese becomes the fixed MLP setting in common.py.
Usage: python src/mlp_es_select.py -> results/mlp_es_select.{json,md}
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from common import RESULTS, SEEDS, load, pipe, save

out, rows = {}, []
for lang in ["su", "id"]:
    trv = pd.concat([load(lang, "train"), load(lang, "valid")], ignore_index=True)
    for feat in ["word_tfidf", "char_tfidf"]:
        for es in [True, False]:
            f1 = []
            for s in SEEDS:
                p = pipe(feat, "mlp", s).set_params(clf__early_stopping=es)
                cv = StratifiedKFold(5, shuffle=True, random_state=s)
                f1.append(cross_val_score(p, trv.text, trv.label, scoring="f1_macro", cv=cv).mean())
            out[f"{lang}/{feat}/es={es}"] = {"cv_macro_f1_mean": float(np.mean(f1)), "cv_macro_f1_std": float(np.std(f1)),
                                             "per_seed": [float(x) for x in f1]}
            rows.append(f"| {lang} | {feat} | {es} | {np.mean(f1):.3f} ± {np.std(f1):.3f} |")
            print(rows[-1], flush=True)
su_mean = {es: np.mean([out[f"su/{f}/es={es}"]["cv_macro_f1_mean"] for f in ["word_tfidf", "char_tfidf"]]) for es in [True, False]}
out["chosen_early_stopping"] = bool(max(su_mean, key=su_mean.get))
out["criterion"] = "mean of su word_tfidf and char_tfidf CV macro-F1 (train+valid, 5-fold, seeds 0-4); test set not used"
save("mlp_es_select", out)
(RESULTS / "mlp_es_select.md").write_text(
    "MLP early_stopping chosen by stratified 5-fold CV on train+valid (test set not used)\n\n"
    "| data | features | early_stopping | CV macro-F1 (mean ± std over seeds) |\n|---|---|---|---|\n"
    + "\n".join(rows) + f"\n\nChosen (Sundanese CV): early_stopping={out['chosen_early_stopping']}\n")
print("chosen early_stopping =", out["chosen_early_stopping"])
