Paired on 400 su test sentences; bootstrap B=2000 per seed, pooled over seeds 0-4; exact McNemar.

| comparison (A vs B) | dF1 (A-B) | 95% CI | McNemar p (median) | seeds p<.05 |
|---|---|---|---|---|
| svm vs rf (su->su) | +0.006 | [-0.044, +0.055] | 0.21 | 0/5 |
| su->su vs id->su (svm) | +0.112 | [+0.053, +0.168] | 1.8e-06 | 5/5 |
| su->su vs id->su (nb) | +0.066 | [+0.014, +0.116] | 7.4e-05 | 5/5 |
| su->su vs id->su (rf) | +0.560 | [+0.510, +0.608] | 1.3e-40 | 5/5 |
| su->su vs id->su (mlp) | +0.071 | [+0.003, +0.138] | 0.00037 | 5/5 |
