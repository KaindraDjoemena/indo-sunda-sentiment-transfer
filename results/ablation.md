| model | features | norm | CV macro-F1 | test macro-F1 (mean ± std) | test acc | best params |
|---|---|---|---|---|---|---|
| svm | word_tfidf | raw | 0.693 | 0.707 ± 0.000 | 0.735 | {'clf__C': '1'} |
| svm | word_tfidf | lower | 0.721 | 0.738 ± 0.000 | 0.762 | {'clf__C': '1'} |
| svm | word_tfidf | norm | 0.732 | 0.738 ± 0.000 | 0.762 | {'clf__C': '1'} |
| svm | word_bow | raw | 0.749 | 0.725 ± 0.000 | 0.733 | {'clf__C': '1'} |
| svm | word_bow | lower | 0.771 | 0.736 ± 0.000 | 0.740 | {'clf__C': '0.1'} |
| svm | word_bow | norm | 0.773 | 0.736 ± 0.000 | 0.740 | {'clf__C': '0.1'} |
| svm | char_tfidf | raw | 0.757 | 0.758 ± 0.000 | 0.770 | {'clf__C': '1'} |
| svm | char_tfidf | lower | 0.764 | 0.776 ± 0.000 | 0.787 | {'clf__C': '1'} |
| svm | char_tfidf | norm | 0.771 | 0.764 ± 0.000 | 0.775 | {'clf__C': '0.1'} |
| svm | char_bow | raw | 0.705 | 0.683 ± 0.000 | 0.698 | {'clf__C': '0.1'} |
| svm | char_bow | lower | 0.710 | 0.713 ± 0.000 | 0.725 | {'clf__C': '0.1'} |
| svm | char_bow | norm | 0.699 | 0.685 ± 0.000 | 0.698 | {'clf__C': '0.1'} |
| nb | word_tfidf | raw | 0.661 | 0.712 ± 0.000 | 0.750 | {'clf__alpha': '0.01'} |
| nb | word_tfidf | lower | 0.684 | 0.699 ± 0.000 | 0.738 | {'clf__alpha': '0.01'} |
| nb | word_tfidf | norm | 0.677 | 0.681 ± 0.000 | 0.728 | {'clf__alpha': '0.01'} |
| nb | word_bow | raw | 0.675 | 0.715 ± 0.000 | 0.750 | {'clf__alpha': '0.1'} |
| nb | word_bow | lower | 0.700 | 0.713 ± 0.000 | 0.743 | {'clf__alpha': '0.1'} |
| nb | word_bow | norm | 0.696 | 0.701 ± 0.000 | 0.735 | {'clf__alpha': '0.1'} |
| nb | char_tfidf | raw | 0.687 | 0.692 ± 0.000 | 0.723 | {'clf__alpha': '0.01'} |
| nb | char_tfidf | lower | 0.695 | 0.716 ± 0.000 | 0.740 | {'clf__alpha': '0.01'} |
| nb | char_tfidf | norm | 0.684 | 0.695 ± 0.000 | 0.720 | {'clf__alpha': '0.01'} |
| nb | char_bow | raw | 0.710 | 0.731 ± 0.000 | 0.750 | {'clf__alpha': '0.1'} |
| nb | char_bow | lower | 0.716 | 0.721 ± 0.000 | 0.740 | {'clf__alpha': '0.1'} |
| nb | char_bow | norm | 0.705 | 0.719 ± 0.000 | 0.735 | {'clf__alpha': '0.1'} |
| rf | word_tfidf | raw | 0.711 | 0.716 ± 0.004 | 0.725 | {'clf__min_samples_leaf': '1'} |
| rf | word_tfidf | lower | 0.717 | 0.735 ± 0.003 | 0.743 | {'clf__min_samples_leaf': '1'} |
| rf | word_tfidf | norm | 0.716 | 0.729 ± 0.010 | 0.738 | {'clf__min_samples_leaf': '1'} |
| rf | word_bow | raw | 0.694 | 0.709 ± 0.009 | 0.717 | {'clf__min_samples_leaf': '2'} |
| rf | word_bow | lower | 0.703 | 0.740 ± 0.003 | 0.748 | {'clf__min_samples_leaf': '1'} |
| rf | word_bow | norm | 0.706 | 0.732 ± 0.007 | 0.740 | {'clf__min_samples_leaf': '1'} |
| rf | char_tfidf | raw | 0.735 | 0.726 ± 0.009 | 0.734 | {'clf__min_samples_leaf': '1'} |
| rf | char_tfidf | lower | 0.744 | 0.730 ± 0.003 | 0.738 | {'clf__min_samples_leaf': '2'} |
| rf | char_tfidf | norm | 0.739 | 0.714 ± 0.004 | 0.722 | {'clf__min_samples_leaf': '2'} |
| rf | char_bow | raw | 0.721 | 0.728 ± 0.009 | 0.737 | {'clf__min_samples_leaf': '1'} |
| rf | char_bow | lower | 0.726 | 0.743 ± 0.008 | 0.750 | {'clf__min_samples_leaf': '1'} |
| rf | char_bow | norm | 0.730 | 0.738 ± 0.005 | 0.747 | {'clf__min_samples_leaf': '1'} |
| mlp | word_tfidf | raw | 0.705 | 0.720 ± 0.008 | 0.750 | {'clf__alpha': '0.0001', 'clf__hidden_layer_sizes': '(128,)'} |
| mlp | word_tfidf | lower | 0.743 | 0.722 ± 0.006 | 0.752 | {'clf__alpha': '0.0001', 'clf__hidden_layer_sizes': '(64,)'} |
| mlp | word_tfidf | norm | 0.738 | 0.699 ± 0.007 | 0.739 | {'clf__alpha': '0.0001', 'clf__hidden_layer_sizes': '(128,)'} |
| mlp | word_bow | raw | 0.724 | 0.736 ± 0.005 | 0.756 | {'clf__alpha': '0.0001', 'clf__hidden_layer_sizes': '(64,)'} |
| mlp | word_bow | lower | 0.756 | 0.740 ± 0.005 | 0.762 | {'clf__alpha': '0.01', 'clf__hidden_layer_sizes': '(128,)'} |
| mlp | word_bow | norm | 0.764 | 0.732 ± 0.015 | 0.755 | {'clf__alpha': '0.01', 'clf__hidden_layer_sizes': '(64,)'} |
| mlp | char_tfidf | raw | 0.735 | 0.744 ± 0.004 | 0.761 | {'clf__alpha': '0.01', 'clf__hidden_layer_sizes': '(128,)'} |
| mlp | char_tfidf | lower | 0.721 | 0.749 ± 0.005 | 0.768 | {'clf__alpha': '0.01', 'clf__hidden_layer_sizes': '(128,)'} |
| mlp | char_tfidf | norm | 0.728 | 0.731 ± 0.005 | 0.752 | {'clf__alpha': '0.01', 'clf__hidden_layer_sizes': '(128,)'} |
| mlp | char_bow | raw | 0.735 | 0.746 ± 0.006 | 0.757 | {'clf__alpha': '0.0001', 'clf__hidden_layer_sizes': '(128,)'} |
| mlp | char_bow | lower | 0.722 | 0.747 ± 0.004 | 0.756 | {'clf__alpha': '0.01', 'clf__hidden_layer_sizes': '(128,)'} |
| mlp | char_bow | norm | 0.735 | 0.725 ± 0.012 | 0.739 | {'clf__alpha': '0.01', 'clf__hidden_layer_sizes': '(64,)'} |
