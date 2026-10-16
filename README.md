# Indonesian-Sundanese Cross-Lingual Transfer for Sentiment Classification with Classical Models

Code and results for the paper of the same title.
The experiments train classical text classifiers on Indonesian sentiment data and test them on Sundanese.
The data is NusaX-senti [1].

Research questions:

- **RQ1, transfer.** Train on Indonesian and test on Sundanese (id->su, zero-shot). Compare with Sundanese-only training (su->su) and with mixed training (Indonesian + k% Sundanese).
- **RQ2, feature ablation.** Word vs character n-grams, TF-IDF vs bag-of-words, and three preprocessing levels, for each model.
- **RQ3, learning curve and significance.** Train on 10/25/50/100% of the Sundanese training data. Compare models with paired tests (exact McNemar, paired bootstrap). Count the errors on the Neutral class.

Models (scikit-learn): linear SVM, multinomial Naive Bayes, Random Forest and an MLP with one hidden layer.

## Main result

Macro-F1 on the Sundanese test set, word TF-IDF features, mean ± std over seeds 0--4 (from `results/transfer.md`):

| setting | SVM | Naive Bayes | Random Forest | MLP |
|---|---|---|---|---|
| su->su | 0.722 ± 0.000 | 0.682 ± 0.000 | 0.716 ± 0.007 | 0.715 ± 0.024 |
| id->su (zero-shot) | 0.611 ± 0.000 | 0.616 ± 0.000 | 0.156 ± 0.003 | 0.644 ± 0.006 |
| id + 10% su (50 sentences) | 0.683 ± 0.017 | 0.598 ± 0.022 | 0.450 ± 0.058 | 0.660 ± 0.024 |

All other numbers are in `results/`.

## Requirements

- Python 3 and `make`. We used Python 3.14.8 on Linux.
- `curl`, to download the data.
- No GPU.

The Python packages are in `requirements.txt` (numpy, pandas, scikit-learn, scipy, matplotlib).
The versions are pinned to the versions that produced `results/`.

## Setup

```bash
git clone <this repository>
cd <repository folder>
make setup    # creates .venv and installs requirements.txt
make data     # downloads NusaX-senti (Indonesian and Sundanese) into data/raw/
```

If your Python command is not `python3`, give it to make: `make setup PYTHON=python`.

Without `make` (for example on Windows), run these commands:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip install -r requirements.txt
```

Then download the data by hand.
Get the files `train.csv`, `valid.csv` and `test.csv` from
`datasets/sentiment/indonesian/` and `datasets/sentiment/sundanese/` in
<https://github.com/IndoNLP/nusax>.
Put them here:

```
data/raw/indonesian/{train,valid,test}.csv
data/raw/sundanese/{train,valid,test}.csv
```

Each language has 500 training, 100 validation and 400 test sentences with the labels negative, neutral and positive.
We use the official split and do not change the files.

## Run the experiments

Run everything, in the correct order:

```bash
make all
```

Or run one step at a time. Each step writes `results/<name>.json` and, for most steps, a table in `results/<name>.md`.

| command | script | output in `results/` | what it does |
|---|---|---|---|
| `make baseline` | `src/baseline.py` | `baseline.json` | su->su and id->su, word TF-IDF |
| `make transfer` | `src/transfer.py` | `transfer.*` | RQ1 with word features: su->su, id->su, mixed k = 10/25/50/100%; vocabulary overlap |
| `make transfer_char` | `src/transfer_char.py` | `transfer_char.*` | RQ1 with character features |
| `make curve` | `src/curve.py` | `curve.*`, `figs/curve.png` | RQ3 learning curve |
| `make curve_noes` | `src/curve_noes.py` | `curve_mlp_compare.*` | learning curve of the MLP with and without early stopping |
| `make mlp_select` | `src/mlp_es_select.py` | `mlp_es_select.*` | chooses the MLP early-stopping setting by cross-validation |
| `make ablation` | `src/ablation.py` | `ablation.*` | RQ2: 4 models x 4 feature types x 3 preprocessing levels |
| `make rf_nocw` | `src/rf_nocw.py` | `rf_nocw.json` | Random Forest id->su with and without class weights |
| `make stats` | `src/stats.py` | `stats.*` | paired tests for RQ1 and RQ3 (needs `baseline.json`) |
| `make stats_extra` | `src/stats_extra.py` | `stats_extra.*` | more paired tests; Holm correction over all tests (needs `stats.json`) |
| `make diagnostics` | `src/diagnostics.py` | `diagnostics.*` | label counts, constant-predictor baseline, known Indonesian features per Sundanese test sentence, Random Forest id->su predictions, MLP iteration counts |
| `make error_analysis` | `src/error_analysis_train.py` | `error_analysis.*` | id->su confusion matrices and per-class F1 |
| `make error_quant` | `src/error_quant.py` | `error_quant.*` | id->su error rate by the number of known Indonesian words per sentence |
| `make figures` | `src/figures.py` | `figs/*.png` | paper figures from the result files (no training) |

`make ablation` is the slowest step. To run its cross-validation folds in parallel, set `N_JOBS`, for example `make ablation N_JOBS=8`.
To rerun only some models, run `.venv/bin/python src/ablation.py svm nb`. The script then merges the new rows into the existing `ablation.json`.

## Method notes

- **Seeds.** Every experiment runs with seeds 0, 1, 2, 3 and 4. The tables write the mean and the standard deviation over the seeds as `mean ± std`. SVM and Naive Bayes are deterministic, so their standard deviation is often 0.
- **Test set.** Hyperparameters are chosen by stratified 5-fold cross-validation on train + valid. The test set is used only for the final evaluation. RQ1 and RQ3 use the fixed defaults in `src/common.py`.
- **Features.** Word n-grams (1--2) and character n-grams (`char_wb`, 2--5), each with TF-IDF or counts (`src/common.py`).
- **Preprocessing levels** (RQ2, `src/ablation.py`): `raw` (no change), `lower` (lowercase) and `norm` (lowercase, remove URLs, @-mentions, punctuation and digits, and shorten repeated characters).
- **Threads.** The Makefile sets `OPENBLAS_NUM_THREADS=1`. One BLAS thread per process gives the same results and avoids slow runs when several processes share the CPU.

## Repository layout

```
src/        experiment scripts; src/common.py has the data loader, features, models and seeds
results/    JSON and Markdown results of every script, figures in results/figs/
data/raw/   NusaX-senti CSV files (not in the repository; download with make data)
```

## Data licence and citation

NusaX is licensed CC-BY-SA (dataset) and Apache-2.0 (code), see <https://github.com/IndoNLP/nusax>.
This repository does not contain the data.
If you use the data, cite NusaX:

```bibtex
@inproceedings{winata-etal-2023-nusax,
    title = "NusaX: Multilingual Parallel Sentiment Dataset for 10 Indonesian Local Languages",
    author = "Winata, Genta Indra  and Aji, Alham Fikri  and Cahyawijaya, Samuel  and Mahendra, Rahmad  and
      Koto, Fajri  and Romadhony, Ade  and Kurniawan, Kemal  and Moeljadi, David  and Prasojo, Radityo Eko  and
      Fung, Pascale  and Baldwin, Timothy  and Lau, Jey Han  and Sennrich, Rico  and Ruder, Sebastian",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.eacl-main.57",
    pages = "815--834"
}
```

[1] G. I. Winata et al., "NusaX: Multilingual Parallel Sentiment Dataset for 10 Indonesian Local Languages," EACL 2023.
