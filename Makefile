# HERON: the common entry points. See README.md and docs/REPRODUCE.md.
PY ?= python

.PHONY: check figures test test-cpu pipeline validate smoke clean

check:            ## verify the installation and the weights
	$(PY) -m heron.check

test-cpu:         ## published numbers from the stored products (no GPU)
	$(PY) -m pytest tests/test_products.py tests/test_core.py tests/test_cli.py tests/test_orchestration.py -q

test:             ## full test suite (GPU regression tests included)
	$(PY) -m pytest tests -q

figures:          ## level 0: redraw the figures from the stored products (no GPU)
	$(PY) evaluation/fit_recalibration.py
	$(PY) evaluation/fig_calibration_summary.py
	$(PY) evaluation/fig_robust_panel.py
	$(PY) evaluation/fig_countdown_twotier.py
	$(PY) evaluation/fig_synergy_mollweide.py
	$(PY) evaluation/fig_synergy_sky.py
	$(PY) evaluation/fig_pointing_ensemble.py --hyp-only
	$(PY) evaluation/fig_two_tier_final.py --alert handoff
	$(PY) evaluation/fig_pp_kref_compare.py
	@echo "figures written to results/figures/"

pipeline:         ## the Sangria Case-1 event through stages 1-3
	$(PY) -m heron.pipeline --simulate --case 1 --n-ucb 30 --out runs/case1

validate:         ## level 1: every validation from the released weights (hours)
	$(PY) -m heron.run --config config/heron.yaml --stage infer,validate --weights released

smoke:            ## level 2 at toy scale: data, training, calibration, validation (minutes)
	$(PY) -m heron.run --config config/smoke.yaml --stage all

clean:
	rm -rf results runs/smoke runs/case1
