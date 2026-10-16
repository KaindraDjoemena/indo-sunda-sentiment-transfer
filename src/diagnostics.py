"""Descriptive diagnostics used in the paper text (no tuning, fixed defaults from common.py).
- label counts per language and split
- constant-predictor macro-F1 on su test
- active id-train features per su test sentence (word, char)
- RF id->su (word, balanced): P(class | empty input), predictions split by active-feature count
- MLP early stopping: iterations run (n_iter_) on the su learning-curve subsamples
Usage: python src/diagnostics.py -> results/diagnostics.{json,md}
"""
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from common import RESULTS, SEEDS, load, pipe, save, vectorizer

CLASSES = ["negative", "neutral", "positive"]
out = {}

out["label_counts"] = {f"{l}/{s}": {c: int((load(l, s).label == c).sum()) for c in CLASSES} for l in ["id", "su"] for s in ["train", "valid", "test"]}

id_tr, su_tr, su_te = load("id", "train"), load("su", "train"), load("su", "test")
y = su_te.label.to_numpy()
out["constant_predictor_macro_f1"] = {c: float(f1_score(y, [c] * len(y), average="macro")) for c in CLASSES}

act = {}
for kind in ["word", "char"]:
    vec = vectorizer(f"{kind}_tfidf").fit(id_tr.text)
    n = np.asarray((vec.transform(su_te.text) > 0).sum(1)).ravel()
    act[kind] = n
    out[f"active_id_features_per_su_test_sentence/{kind}"] = dict(
        median=float(np.median(n)), mean=float(n.mean()), n_zero=int((n == 0).sum()), n_sentences=len(n))

rf = {"p_empty": [], "neutral_rate_zero_active": [], "neutral_rate_ge1_active": [], "n_pred_neutral": []}
zero = act["word"] == 0
for s in SEEDS:
    p = pipe("word_tfidf", "rf", s).fit(id_tr.text, id_tr.label)
    cls = list(p.classes_)
    rf["p_empty"].append(dict(zip(cls, p.predict_proba([""])[0].tolist())))
    pred = p.predict(su_te.text)
    rf["n_pred_neutral"].append(int((pred == "neutral").sum()))
    rf["neutral_rate_zero_active"].append(float((pred[zero] == "neutral").mean()))
    rf["neutral_rate_ge1_active"].append(float((pred[~zero] == "neutral").mean()))
out["rf_word_id_su_balanced"] = dict(
    p_empty_mean={c: float(np.mean([d[c] for d in rf["p_empty"]])) for c in CLASSES},
    n_pred_neutral_mean=float(np.mean(rf["n_pred_neutral"])),
    neutral_rate_zero_active_mean=float(np.mean(rf["neutral_rate_zero_active"])),
    neutral_rate_ge1_active_mean=float(np.mean(rf["neutral_rate_ge1_active"])),
    per_seed=rf)

es = {}
for f in [0.10, 0.25, 0.50, 1.00]:
    its = []
    for s in SEEDS:
        sub = su_tr if f == 1.0 else su_tr.iloc[train_test_split(np.arange(len(su_tr)), train_size=f, stratify=su_tr.label, random_state=s)[0]]
        its.append(int(pipe("word_tfidf", "mlp", s).set_params(clf__early_stopping=True).fit(sub.text, sub.label).named_steps["clf"].n_iter_))
    es[f"{int(f*100)}"] = dict(n_train=len(sub), n_iter_per_seed=its, n_iter_median=float(np.median(its)), n_valid_internal=int(round(len(sub) * 0.1)))
out["mlp_early_stopping_n_iter"] = es
save("diagnostics", out)

L = ["# Diagnostics (descriptive; fixed defaults; seeds 0-4)", "", "## Label counts", "", "| split | negative | neutral | positive |", "|---|---|---|---|"]
L += [f"| {k} | {v['negative']} | {v['neutral']} | {v['positive']} |" for k, v in out["label_counts"].items()]
L += ["", "## Constant predictor on su test (macro-F1)", "", "| always | macro-F1 |", "|---|---|"]
L += [f"| {c} | {v:.3f} |" for c, v in out["constant_predictor_macro_f1"].items()]
L += ["", "## Active id-train features per su test sentence", "", "| features | median | mean | sentences with 0 |", "|---|---|---|---|"]
for kind in ["word", "char"]:
    d = out[f"active_id_features_per_su_test_sentence/{kind}"]
    L.append(f"| {kind} | {d['median']:.1f} | {d['mean']:.1f} | {d['n_zero']} of {d['n_sentences']} |")
r = out["rf_word_id_su_balanced"]
L += ["", "## RF id->su, word_tfidf, class_weight=balanced (mean over seeds)", "",
      f"- P(class | empty input): " + ", ".join(f"{c} {v:.3f}" for c, v in r["p_empty_mean"].items()),
      f"- predicted neutral: {r['n_pred_neutral_mean']:.1f} of 400",
      f"- neutral rate, sentences with 0 active features: {r['neutral_rate_zero_active_mean']:.3f}",
      f"- neutral rate, sentences with >=1 active feature: {r['neutral_rate_ge1_active_mean']:.3f}",
      "", "## MLP (early_stopping=True) iterations on su learning-curve subsamples", "",
      "| fraction | n_train | internal valid size | n_iter per seed | median |", "|---|---|---|---|---|"]
L += [f"| {k}% | {v['n_train']} | {v['n_valid_internal']} | {v['n_iter_per_seed']} | {v['n_iter_median']:.0f} |" for k, v in es.items()]
(RESULTS / "diagnostics.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
