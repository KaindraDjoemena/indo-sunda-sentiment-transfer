"""Per-class F1 and confusion matrices for zero-shot id->su transfer (word TF-IDF, seed 0).
Models are trained on the Indonesian train split (500 sentences) and tested on the Sundanese test split.
Usage: python src/error_analysis_train.py -> results/error_analysis.{json,md}
"""
import json

from sklearn.metrics import classification_report, confusion_matrix

from common import MODELS, RESULTS, load, pipe

train_id = load("id", "train")
test_su = load("su", "test")
seed = 0

out = {}
out_lines = [
    "# Error Analysis: Zero-Shot ID -> SU Transfer (Trained on Train Only, n=500, Seed 0)",
    "",
    "| Model | Macro-F1 | Negative F1 | Neutral F1 | Positive F1 | Confusion Matrix (Rows: Neg, Neu, Pos; Cols: Neg, Neu, Pos) |",
    "|---|---|---|---|---|---|",
]

for m in MODELS:
    clf = pipe("word_tfidf", m, seed)
    clf.fit(train_id["text"], train_id["label"])
    preds = clf.predict(test_su["text"])
    report = classification_report(test_su["label"], preds, output_dict=True)
    cm = confusion_matrix(test_su["label"], preds, labels=["negative", "neutral", "positive"])

    out[m] = {
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "accuracy": float(report["accuracy"]),
        "per_class": {
            "negative": float(report["negative"]["f1-score"]),
            "neutral": float(report["neutral"]["f1-score"]),
            "positive": float(report["positive"]["f1-score"]),
        },
        "confusion_matrix": cm.tolist(),
    }
    cm_str = f"[{cm[0].tolist()}, {cm[1].tolist()}, {cm[2].tolist()}]"
    out_lines.append(
        f"| {m.upper()} | {report['macro avg']['f1-score']:.3f} | {report['negative']['f1-score']:.3f} | {report['neutral']['f1-score']:.3f} | {report['positive']['f1-score']:.3f} | `{cm_str}` |"
    )

(RESULTS / "error_analysis.json").write_text(json.dumps(out, indent=2))
(RESULTS / "error_analysis.md").write_text("\n".join(out_lines) + "\n")
print("Saved results/error_analysis.json and results/error_analysis.md")
