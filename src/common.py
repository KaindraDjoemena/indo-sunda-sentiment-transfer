"""Shared helpers: data loading, features, models, metrics."""
import json
import pathlib

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
SEEDS = [0, 1, 2, 3, 4]
LANGS = {"id": "indonesian", "su": "sundanese"}


def load(lang: str, split: str) -> pd.DataFrame:
    """lang in LANGS, split in train/valid/test. Run `make data` first."""
    return pd.read_csv(RAW / LANGS[lang] / f"{split}.csv")


def vectorizer(kind: str = "word_tfidf"):
    """kind: {word,char}_{tfidf,bow}. char uses char_wb 2-5 grams."""
    analyzer, weight = kind.split("_")
    kw = dict(lowercase=True, min_df=1)
    kw.update(analyzer="word", ngram_range=(1, 2)) if analyzer == "word" else kw.update(analyzer="char_wb", ngram_range=(2, 5))
    return TfidfVectorizer(**kw) if weight == "tfidf" else CountVectorizer(**kw)


def model(name: str, seed: int = 0):
    return {
        "svm": LinearSVC(C=1.0, class_weight="balanced", random_state=seed),
        "nb": MultinomialNB(alpha=0.1),
        "rf": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1),
        # early_stopping=False was chosen by 5-fold CV on train+valid (src/mlp_es_select.py, results/mlp_es_select.json)
        "mlp": MLPClassifier(hidden_layer_sizes=(64,), early_stopping=False, max_iter=300, random_state=seed),
    }[name]


MODELS = ["svm", "nb", "rf", "mlp"]


def pipe(feat: str, name: str, seed: int = 0) -> Pipeline:
    return Pipeline([("vec", vectorizer(feat)), ("clf", model(name, seed))])


def score(y_true, y_pred) -> dict:
    return {"macro_f1": f1_score(y_true, y_pred, average="macro"), "acc": accuracy_score(y_true, y_pred)}


def save(name: str, obj: dict):
    """Write results/<name>.json."""
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(obj, indent=2, default=float))
