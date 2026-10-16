# Diagnostics (descriptive; fixed defaults; seeds 0-4)

## Label counts

| split | negative | neutral | positive |
|---|---|---|---|
| id/train | 192 | 119 | 189 |
| id/valid | 38 | 24 | 38 |
| id/test | 153 | 96 | 151 |
| su/train | 192 | 119 | 189 |
| su/valid | 38 | 24 | 38 |
| su/test | 153 | 96 | 151 |

## Constant predictor on su test (macro-F1)

| always | macro-F1 |
|---|---|
| negative | 0.184 |
| neutral | 0.129 |
| positive | 0.183 |

## Active id-train features per su test sentence

| features | median | mean | sentences with 0 |
|---|---|---|---|
| word | 4.5 | 5.6 | 12 of 400 |
| char | 246.0 | 268.3 | 0 of 400 |

## RF id->su, word_tfidf, class_weight=balanced (mean over seeds)

- P(class | empty input): negative 0.021, neutral 0.975, positive 0.004
- predicted neutral: 390.6 of 400
- neutral rate, sentences with 0 active features: 1.000
- neutral rate, sentences with >=1 active feature: 0.976

## MLP (early_stopping=True) iterations on su learning-curve subsamples

| fraction | n_train | internal valid size | n_iter per seed | median |
|---|---|---|---|---|
| 10% | 50 | 5 | [12, 12, 12, 21, 20] | 12 |
| 25% | 125 | 12 | [12, 12, 12, 12, 12] | 12 |
| 50% | 250 | 25 | [29, 27, 12, 12, 18] | 18 |
| 100% | 500 | 50 | [19, 37, 25, 32, 20] | 25 |
