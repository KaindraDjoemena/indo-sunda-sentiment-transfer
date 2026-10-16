# Error Analysis: Zero-Shot ID -> SU Transfer (Trained on Train Only, n=500, Seed 0)

| Model | Macro-F1 | Negative F1 | Neutral F1 | Positive F1 | Confusion Matrix (Rows: Neg, Neu, Pos; Cols: Neg, Neu, Pos) |
|---|---|---|---|---|---|
| SVM | 0.611 | 0.567 | 0.528 | 0.737 | `[[74, 66, 13], [17, 70, 9], [17, 33, 101]]` |
| NB | 0.616 | 0.625 | 0.484 | 0.741 | `[[95, 18, 40], [34, 37, 25], [22, 2, 127]]` |
| RF | 0.158 | 0.000 | 0.387 | 0.087 | `[[0, 152, 1], [0, 94, 2], [0, 144, 7]]` |
| MLP | 0.636 | 0.605 | 0.565 | 0.738 | `[[91, 41, 21], [26, 59, 11], [31, 13, 107]]` |
