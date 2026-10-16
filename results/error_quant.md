id->su, word_tfidf, mean over seeds 0-4. Active Indonesian word features per su test sentence: median 4.5; low = at or below the median (n = 200, Neutral 55), high = above (n = 200, Neutral 41).

| model | error, all, low | error, all, high | error, Neutral, low | error, Neutral, high |
|---|---|---|---|---|
| svm | 0.465 | 0.310 | 0.291 | 0.244 |
| nb | 0.415 | 0.290 | 0.673 | 0.537 |
| rf | 0.725 | 0.771 | 0.000 | 0.044 |
| mlp | 0.386 | 0.307 | 0.385 | 0.424 |
