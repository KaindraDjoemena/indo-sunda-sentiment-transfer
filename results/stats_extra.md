Paired on 400 su test sentences; fixed defaults (common.py); bootstrap B=2000 per seed, pooled over seeds 0-4; exact McNemar per seed; Holm correction over the 17 median p-values.

| comparison (A vs B) | F1 A | F1 B | dF1 (A-B) | 95% CI | McNemar p (median) | Holm p | seeds p<.05 |
|---|---|---|---|---|---|---|---|
| (a) char vs word, id->su (svm) | 0.627 | 0.611 | +0.016 | [-0.038, +0.070] | 0.66 | 1 | 0/5 |
| (a) char vs word, id->su (nb) | 0.589 | 0.616 | -0.028 | [-0.082, +0.024] | 0.92 | 1 | 0/5 |
| (a) char vs word, id->su (rf) | 0.459 | 0.156 | +0.302 | [+0.245, +0.361] | 3.2e-19 | 5.1e-18 | 5/5 |
| (a) char vs word, id->su (mlp) | 0.614 | 0.644 | -0.030 | [-0.089, +0.028] | 0.72 | 1 | 0/5 |
| (b) char vs word, su->su (svm) | 0.742 | 0.722 | +0.020 | [-0.023, +0.064] | 1 | 1 | 0/5 |
| (b) char vs word, su->su (nb) | 0.617 | 0.682 | -0.065 | [-0.110, -0.019] | 0.0018 | 0.023 | 5/5 |
| (b) char vs word, su->su (rf) | 0.721 | 0.716 | +0.005 | [-0.040, +0.055] | 0.79 | 1 | 0/5 |
| (b) char vs word, su->su (mlp) | 0.704 | 0.715 | -0.011 | [-0.081, +0.057] | 0.55 | 1 | 0/5 |
| (c) mixed_k10 vs id->su, word (svm) | 0.683 | 0.611 | +0.072 | [+0.010, +0.138] | 0.0015 | 0.021 | 5/5 |
| (c) mixed_k10 vs id->su, word (nb) | 0.598 | 0.616 | -0.019 | [-0.079, +0.053] | 0.29 | 1 | 1/5 |
| (c) mixed_k10 vs id->su, word (rf) | 0.450 | 0.156 | +0.293 | [+0.205, +0.421] | 1.3e-23 | 2.2e-22 | 5/5 |
| (c) mixed_k10 vs id->su, word (mlp) | 0.660 | 0.644 | +0.016 | [-0.049, +0.081] | 0.14 | 1 | 2/5 |
| (d) mixed_k10 vs su-only 10%, word (svm) | 0.683 | 0.551 | +0.132 | [+0.057, +0.202] | 0.00091 | 0.014 | 5/5 |
| (d) mixed_k10 vs su-only 10%, word (nb) | 0.598 | 0.529 | +0.069 | [-0.046, +0.174] | 0.31 | 1 | 2/5 |
| (d) mixed_k10 vs su-only 10%, word (rf) | 0.450 | 0.589 | -0.140 | [-0.268, -0.009] | 0.0081 | 0.098 | 3/5 |
| (d) mixed_k10 vs su-only 10%, word (mlp) | 0.660 | 0.573 | +0.087 | [-0.037, +0.202] | 0.06 | 0.6 | 2/5 |
| (e) RF id->su word: class_weight None vs balanced | 0.199 | 0.156 | +0.042 | [+0.013, +0.080] | 0.021 | 0.24 | 3/5 |
