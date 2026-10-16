MLP early_stopping chosen by stratified 5-fold CV on train+valid (test set not used)

| data | features | early_stopping | CV macro-F1 (mean ± std over seeds) |
|---|---|---|---|
| su | word_tfidf | True | 0.577 ± 0.043 |
| su | word_tfidf | False | 0.728 ± 0.012 |
| su | char_tfidf | True | 0.665 ± 0.019 |
| su | char_tfidf | False | 0.719 ± 0.015 |
| id | word_tfidf | True | 0.641 ± 0.031 |
| id | word_tfidf | False | 0.737 ± 0.012 |
| id | char_tfidf | True | 0.680 ± 0.041 |
| id | char_tfidf | False | 0.729 ± 0.009 |

Chosen (Sundanese CV): early_stopping=False
