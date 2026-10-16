# Reproduce all experiments: make setup && make data && make all
PYTHON ?= python3
PY = .venv/bin/python
N_JOBS ?= 1
# One BLAS thread per process: the results are the same and the run is faster than with oversubscribed threads.
export OPENBLAS_NUM_THREADS = 1
export OMP_NUM_THREADS = 1
export MKL_NUM_THREADS = 1

B = https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/sentiment
EXPERIMENTS = baseline transfer transfer_char curve curve_noes mlp_select ablation rf_nocw \
              stats stats_extra diagnostics error_analysis error_quant

.PHONY: setup data all experiments figures $(EXPERIMENTS)

setup:
	$(PYTHON) -m venv .venv && $(PY) -m pip install -r requirements.txt

data:
	@for l in indonesian sundanese; do mkdir -p data/raw/$$l; \
	  for s in train valid test; do curl -sfL $(B)/$$l/$$s.csv -o data/raw/$$l/$$s.csv || exit 1; done; done
	@wc -l data/raw/*/*.csv

# Runs the experiments one after another (stats needs baseline, stats_extra needs stats), then the figures.
all:
	@for t in $(EXPERIMENTS) figures; do $(MAKE) --no-print-directory $$t || exit 1; done

experiments:
	@for t in $(EXPERIMENTS); do $(MAKE) --no-print-directory $$t || exit 1; done

baseline:       ; $(PY) src/baseline.py
transfer:       ; $(PY) src/transfer.py
transfer_char:  ; $(PY) src/transfer_char.py
curve:          ; $(PY) src/curve.py
curve_noes:     ; $(PY) src/curve_noes.py
mlp_select:     ; $(PY) src/mlp_es_select.py
ablation:       ; N_JOBS=$(N_JOBS) $(PY) src/ablation.py
rf_nocw:        ; $(PY) src/rf_nocw.py
stats:          ; $(PY) src/stats.py
stats_extra:    ; $(PY) src/stats_extra.py
diagnostics:    ; $(PY) src/diagnostics.py
error_analysis: ; $(PY) src/error_analysis_train.py
error_quant:    ; $(PY) src/error_quant.py
figures:        ; $(PY) src/figures.py

-include paper.mk
