# evaluation/: validations and figures

Each script documents its inputs and outputs in its docstring. Numerical
products are written to `results/products/` and figures to `results/figures/`.
`docs/REPRODUCE.md` maps every figure, table and number of the paper to a
script.

## Computations (GPU, released weights)

| script | result |
|---|---|
| `pp_marginalized.py` | percentile ranks and sky credibility of the truth over a population (calibration) |
| `fit_recalibration.py` | calibration map from the calibration set; recalibrated test ranks; Fig. 3a |
| `eval_skygnpe.py` | 90% sky area per pose iteration and by SNR on the test set |
| `eval_robust.py` | sky area against the number of resolved UCBs |
| `countdown.py` | sky area against time to merger: population and the Case-1 track |
| `sangria_marginalized.py` | alert posterior against the exact MCMC posterior for the Sangria cases |
| `eval_ucb_flow.py` | UCB flow calibration, frequency precision, throughput |
| `ucb_subtract_test.py` | residual after F-statistic subtraction (`--fstat`) |
| `calibrate_ucb.py` | calibrated UCB line response used by the foreground augmentation |
| `recon_sky_test.py` | coherent-grid refinement on emulated reconstructions (projection) |
| `resolved_population_1month.py` | one-month resolved population of the Sangria catalogue |
| `probe_oracle_ref.py`, `probe_spin_het.py`, `refine_infer.py`, `is_refine.py` | Supplementary Note on the reference-echo limit: reference offset sweep, spin-reference retention, self-refinement, importance sampling |

## Figures (from products; no GPU unless noted)

| script | figure |
|---|---|
| `fig_two_tier_final.py` | Fig. 2 (GPU for `--alert recompute`) |
| `fit_recalibration.py`, `fig_robust_panel.py`, `eval_ucb_flow.py` | Fig. 3 |
| `fig_countdown_twotier.py` | Fig. 4 |
| `fig_synergy_mollweide.py`, `fig_synergy_sky.py`, `fig_pointing_ensemble.py --hyp-only` | Fig. 5 |
| `fig_fullband_scan.py` | full-band scan of the realistic population (GPU) |
| `fig_calibration_summary.py` | Extended Data Fig. 4 |
| `fig_ucb_recovery.py` | Extended Data Fig. 5 (GPU) |
| `fig_pp_kref_compare.py` | reference pooling against a single reference |

`plotstyle.py` holds the shared matplotlib style.
