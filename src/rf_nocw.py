"""Random Forest, zero-shot id->su: class_weight="balanced" vs None, word and char TF-IDF.
Checks whether the class weights cause the many neutral predictions.
Usage: python src/rf_nocw.py -> results/rf_nocw.json
"""
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from common import RESULTS, SEEDS, load, score, vectorizer

id_tr = load("id", "train")
su_te = load("su", "test")

results = {}
for feat in ["word_tfidf", "char_tfidf"]:
    results[feat] = {}
    vec = vectorizer(feat)
    X_tr = vec.fit_transform(id_tr.text)
    X_te = vec.transform(su_te.text)

    for cw in ["balanced", None]:
        cw_name = "balanced" if cw == "balanced" else "none"
        f1s, accs, preds_summary = [], [], {}
        for s in SEEDS:
            rf = RandomForestClassifier(n_estimators=300, random_state=s, class_weight=cw)
            rf.fit(X_tr, id_tr.label)
            preds = rf.predict(X_te)
            sc = score(su_te.label, preds)
            f1s.append(sc["macro_f1"])
            accs.append(sc["acc"])

            unique, counts = np.unique(preds, return_counts=True)
            for u, c in zip(unique, counts):
                preds_summary[u] = preds_summary.get(u, 0) + c

        avg_preds = {k: v / len(SEEDS) for k, v in preds_summary.items()}
        results[feat][cw_name] = {
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s)),
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "avg_predictions": avg_preds
        }
        print(f"[{feat}] class_weight={cw_name}: F1={np.mean(f1s):.3f} ± {np.std(f1s):.3f}, preds={avg_preds}")

(RESULTS / "rf_nocw.json").write_text(json.dumps(results, indent=2))
print("Saved to results/rf_nocw.json")
