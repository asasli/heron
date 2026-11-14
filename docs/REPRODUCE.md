# Reproducing the paper

For every figure, table and quoted number: the command, the stored product, the
published value and the value obtained when this release was rerun from the
released weights (RTX A6000, environment of `environment.yml`, November 2026).

Conventions used below.

- *Product*: a file in `paper_products/` (published) or `results/products/`
  (your rerun). Scripts read your rerun if it exists and the published file
  otherwise.
- *Exact*: the rerun reproduces the stored values to floating-point rounding.
- *Equivalent*: the rerun analyses the same simulated events with new flow
  draws, or new noise realizations; the values agree within sampling noise.
  The published runs were made before the flow draws were seeded. In this
  release every script is seeded, so two reruns on the same hardware agree.

## Level 0: figures from the stored products (no GPU)

```bash
make figures
pytest tests/test_products.py tests/test_core.py
```

`make figures` redraws Fig. 2 (hand-off alert), Fig. 3a, Fig. 3 robustness
panel, Fig. 4, Fig. 5a,b,c, Extended Data Fig. 4 and the reference-pooling
comparison into `results/figures/`. `tests/test_products.py` asserts the quoted
numbers against the stored products.

## Level 1: from the released weights

```bash
heron check
heron infer --case 1 --n-ucb 30 --out runs/case1
heron validate                       # python -m heron.run --config config/heron.yaml --stage infer,validate --weights released
python -m yolo.validate_fullband
python -m yolo.detection_metrics
```

## Level 2: training

```bash
heron train --smoke      # toy scale, 6 to 12 min    (python -m heron.run --config config/smoke.yaml --stage all)
heron train              # published configuration  (python -m heron.run --config config/heron.yaml --stage all)
heron infer --weights runs/paper/weights --out runs/mine     # the pipeline on the retrained bundle
```

See `docs/TRAINING_PROVENANCE.md` for what a retraining reproduces.

## Figure by figure

| item | command | product | published | rerun | status |
|---|---|---|---|---|---|
| Fig. 2, alert against stage 4, 68% intervals | `evaluation/fig_two_tier_final.py --n-ucb 30` | `pointing/synergy_subtracted_alert.npz`, `synergy_subtracted_none.npz` | Mc [3.908, 3.918], q [1.178, 1.234], chi_eff [-0.346, -0.318], log D [4.746, 5.018], cos i [-0.122, 0.327] | Mc [3.908, 3.918], q [1.178, 1.233], chi_eff [-0.346, -0.318], log D [4.745, 5.020], cos i [-0.127, 0.321] | equivalent |
| Fig. 2 with the pipeline's own alert | `evaluation/fig_two_tier_final.py --alert handoff` | `handoff/sangria_synergy_subtracted.npz` | not in the paper | Mc [3.908, 3.918], q [1.177, 1.229], chi_eff [-0.365, -0.334], log D [4.744, 5.020], cos i [-0.137, 0.378] | see note 1 |
| Fig. 3a, KS p-values of the 11 recalibrated marginals | `evaluation/pp_marginalized.py --n-events 500 --ref-scale 1.0`, then `evaluation/fit_recalibration.py --test results/products/pp_ranks.npz` | `pp_ranks.npz`, `pp_ranks_cal.npz`, `pp_ranks_recal.npz` | 0.08 to 0.94 | 0.14 to 0.83 | equivalent; exact from the stored ranks |
| Calibration map | `evaluation/fit_recalibration.py` | `pp_ranks_cal.npz` | `weights/recal_map.npz` | byte-identical map | exact |
| Sky coverage of the 50% / 90% credible regions | `evaluation/pp_marginalized.py --n-events 500 --ref-scale 0.45` | `pp_ranks_skycov.npz` | 55.2% / 89.8% | 57.0% / 90.0% at scale 0.45; 53.8% / 89.8% at scale 1.0 | equivalent; note 2 |
| Extended Data Fig. 4, bias and coverage | `evaluation/fig_calibration_summary.py` | `pp_ranks_recal.npz` | 50% coverage 0.47 to 0.53, 90% coverage 0.89 to 0.92 | identical | exact |
| Fig. 3, UCB flow P-P; frequency precision; throughput | `evaluation/eval_ucb_flow.py` | test split `data/ucb/test` | 130 nHz; 12.7 ms per source | 130.2 nHz; 18.6 ms per source | equivalent; note 3 |
| Fig. 3, robustness to the foreground | `evaluation/eval_robust.py --rounds 4 --n-events 250 --data data/mbhb_ridge`, `fig_robust_panel.py` | `robust_sweep.npz` | median 90% area 2278 / 2307 / 2238 / 2468 deg² at 0 / 10 / 20 / 40 UCBs | 2087 / 2263 / 2216 / 2451 deg²; per-event ratio to the stored areas 0.99 to 1.01 (median) | equivalent; note 9 |
| Sky area by SNR (population) | `evaluation/eval_skygnpe.py --rounds 4 --stride 4 --data data/mbhb_ridge` | `skygnpe_eval.npz` | median 1654 deg² (500 events); 3531 / 1702 / 1247 / 1024 deg² for SNR 10-50 / 50-200 / 200-1000 / above 1000 | median 1662 deg²; 3543 / 1834 / 1242 / 967 deg²; per-event ratio to the stored areas 1.00 (median), 0.92 to 1.06 (central 68%) | equivalent; note 9 |
| Fig. 4, countdown | `evaluation/countdown.py`, `fig_countdown_twotier.py` | `countdown_data.npy` | Case-1 track 19714 / 20215 / 19083 / 17679 / 16621 / 8431 deg² at 600 / 300 / 100 / 24 / 6 / 1 h | 19760 / 19756 / 18894 / 18742 / 17357 / 8101 deg²; population medians per bin within 2% of the stored ones | equivalent |
| Fig. 5a,b, stage-4 areas (90% / 50%) and dominant-mode probability | `evaluation/fig_synergy_sky.py`, `fig_synergy_mollweide.py` | `pointing/synergy_*.npz` | subtracted 143 / 28, 0.81; foreground 134 / 32, 0.53; clean 191 / 41, 0.72; unseeded 772 / 109 | identical | exact from the stored chains; note 4 |
| Fig. 5c, ensemble over 4 noise realizations | `evaluation/fig_pointing_ensemble.py --hyp-only` | `pointing_ensemble_canonical.npz`, ensemble chains archive | 180 ± 24, 119 ± 94, 132 ± 82 deg² | identical | exact from the stored chains; note 4 |
| Stage 1 to 3 hand-off of the Case-1 event | `python pointing/export.py` | `handoff/sangria_synergy_*.npz` | the files given to stage 4 | data identical to 1e-8, alert samples to 3e-6 | exact; note 8 |
| Detector test set (Extended Data Fig. 1) | `python -m yolo.detection_metrics` | `data/yolo/*/test` | UCB 4768 true positives, 489 misses; precision / recall 0.98 / 0.91 | UCB 4768 TP, 91 FP, 489 misses (0.981 / 0.907); MBHB 254 TP, 1 FP, 4 misses (0.996 / 0.984) | exact; note 5 |
| Full-band scans (Extended Data Fig. 2) | `python -m yolo.validate_fullband` | `reruns/fullband_validation.json` | 44/44 UCBs, frequency error ≤ 3.4 µHz, no false positive, MBHB median confidence 0.96 | 45/45 UCBs, ≤ 3.5 µHz, 2 detections without counterpart in one of six scans, MBHB 98 tiles at 0.96; dense band 16/20 and 12/13 | equivalent (new noise realizations); note 6 |
| Full-band scan of the realistic population | `evaluation/fig_fullband_scan.py` | `sangria_1month_snr.npz` | 180 of 307 resolved UCBs detected | 182 of 307; MBHB flag not raised in this scan | equivalent; note 6 |
| Resolved population after one month (Methods) | `evaluation/resolved_population_1month.py --catalog LDC2_sangria_training_v2.h5` | `sangria_1month_snr.npz` | 307 above SNR 10 in 0.1-6 mHz, 94 below 4 mHz; per tile median 11, maximum 20 | identical | exact |
| F-statistic subtraction (Methods) | `evaluation/ucb_subtract_test.py --n 150 --fstat` | none | residual SNR 7.6 per source, alpha = 0.14 | 7.5, 0.140 (7.0, 0.133 in a second run; note 8) | equivalent |
| UCB line response | `evaluation/calibrate_ucb.py` | `weights/ucb_response.npz` | released file | identical | exact |
| Coherent-grid projection (Methods) | `evaluation/recon_sky_test.py --case 1` | none | 0.04 to 0.17 deg², 13/13 covered, peak shift 0.03° to 0.27° | 0.034 to 0.168 deg², 13/13, 0.025° to 0.271° | exact |
| Alert against exact MCMC (Table 1) | `evaluation/sangria_marginalized.py --case 1 --k-ref 1 --wide-spin-ref --clean` | `reference_chains/case1_ptmcmc.npz` | agreement on the well-measured parameters | Mc [3.91, 3.92] against [3.88, 3.92]; q [1.18, 1.24] against [1.19, 1.36]; log D [4.74, 5.02] against [4.71, 4.93] | equivalent |
| Reference pooling over-disperses (Supplementary Note) | `evaluation/pp_marginalized.py --k-ref 16 ...`, `fig_pp_kref_compare.py` | `pp_ranks_k16.npz` | rank spread below uniform for the intrinsic parameters | rank standard deviation 0.094 (Mc), 0.134 (q) against 0.289 for uniform | exact from the stored ranks |
| Importance sampling fails (Supplementary Note) | `evaluation/is_refine.py` | `is_refine_case1.npz` | effective sample size 1 of 4096 | effective sample size 1 of 4096 | equivalent |
| Training data | `pe_smbhb/data_mbhb.py --seed 90000 --wide-spin-ref`, `pe_ucb/data_ucb.py --seed 42` | stored first samples in `tests/reference/` | the published training sets | bit-identical | exact |
| Stage timings | `README.md`, "Measured run times" | none | Methods, "Timing" | see note 7 | differs |

## Notes

**1. The alert posterior of Fig. 2.** The published red contours were computed
by `fig_two_tier_final.py` itself: same binary, same instrument-noise
realization as the hand-off, foreground represented by 30 lines injected
through the calibrated response. `--alert handoff` draws instead the alert that
the pipeline produced on the foreground data of the hand-off (30 UCB waveforms
and a confusion realization), which are the draws that seeded the stage-4
chain. The two agree within their widths.

**2. Reference scale.** The heterodyne reference is drawn around the source at
a kernel scale s (`--ref-scale`). The published test set of Fig. 3, the
calibration set and therefore the released calibration map were produced at
s = 1.0. The pipeline alert (`heron/pipeline.py`, the hand-off, Fig. 2) and the
published sky-coverage run use s = 0.45. The calibration map depends on s
through the individual spins. Rerun of the 500 test events with the released
map:

| | s = 1.0 | s = 0.45 |
|---|---|---|
| KS p-value, minimum over the 11 parameters | 0.14 | 0.000 (chi_1), 0.003 (chi_2) |
| KS p-value, the other nine parameters | 0.14 to 0.83 | 0.11 to 0.81 |
| KS p-value, chi_eff | 0.74 | 0.95 |
| sky coverage 50% / 90% | 53.8% / 89.8% | 57.0% / 90.0% |

At s = 1.0 all eleven marginals and the sky regions are calibrated. At s = 0.45
the two individual spins are not calibrated by the released map, while chi_eff,
the other parameters and the sky are. `--ref-scale` is exposed in the pipeline;
the default stays 0.45 so that the published hand-off is reproduced. Products:
`paper_products/reruns/pp_ranks_refscale{1.0,0.45}.npz`.

**3. UCB flow.** The amplitude marginal has a KS p-value below 0.01 on the 4000
test windows in the published run and in the rerun; the other six parameters
are between 0.01 and 0.48.

**4. Stage 4.** The stage-4 sampler was not rerun for this release
(`docs/INSTALL.md`, "Stage 4"); the areas are recomputed from the stored
chains with the canonical estimator. The ensemble chains come from runs with
the hyperbolic and noise-fit Gaussian likelihoods made with scripts of the
HyperWave repository (`docs/SCOPE.md`). From the same chains: the probability
of the island that contains the injection is 0.08, 0.28, 0.00, 0.56 on the
foreground data and 0.66, 0.52, 0.04, 0.74 after subtraction for the four
realizations, and the injection lies inside the 90% region in 4, 2 and 2 of 4
realizations for the clean, foreground and subtracted data.

**5. Detector test set.** The count of UCB false positives is 91. The number
254 is the count of MBHB true positives.

**6. Full-band scans.** The published statements combine several validation
scans (stored catalogues of the original runs: `results/fullband_*` of the
development repository). `yolo/validate_fullband.py` repeats the same set with
fixed seeds. In the scan of the realistic one-month population
(`fig_fullband_scan.py`: 307 resolved binaries and an SNR-357 MBHB with merger
144 h after the end of the segment) the detector returns 182 UCBs and does not
raise the MBHB flag; the chirp track drawn in that figure is the injected one.

**7. Timings.** Measured in this release on one RTX A6000: tiled detection of
71 tiles 1.7 s; one flow evaluation 0.32 s for up to 1,000 samples and 0.38 ms
per additional sample (4,000 samples 1.3 s, 32,000 samples 12.1 s); the alert as
configured in the pipeline (three pose iterations, 32,000 samples) 15.5 s and
in the calibration runs (3,200 samples) 3.2 s; UCB flow 18.6 ms per
source for 1,000 samples; flow plus two-stage F-statistic subtraction 0.19 s per
source (30 sources 5.8 s). Stages 1 to 3 of the Case-1 run take 23 s.

**8. Hand-off.** The clean and foreground data are reproduced to 1e-8 on every
run and the alert samples to single-precision round-off. The subtracted data
are reproduced to 1e-8 on most runs. On a minority of runs made while the GPU
was shared, GPU round-off switched the frequency scan of one source between two
nearly tied grid points, which changes the 128-bin window of that source and
nothing else; `tests/test_gpu_regression.py` allows for it.

**9. Two MBHB test splits.** The stored population products `skygnpe_eval.npz`
and `robust_sweep.npz` were computed with the released weights on the test split
of the `ridge` training set (`data/mbhb_ridge`), the countdown on the test split
of the `wide` set (`data/mbhb`). The reruns in the table use the same splits and
the same events; the remaining differences come from the flow draws, which were
not seeded in the original runs. On the `wide` test split the same commands give
a median 90% area of 1868 deg² over 1000 events (2309 / 1998 / 1668 / 1126 deg²
in the four SNR bins) and 1873 / 1712 / 1936 / 2084 deg² at 0 / 10 / 20 / 40
UCBs. The two splits contain different events, and the difference between their
medians is of the size of the sampling error of a median over 250 to 500 events
(120 to 330 deg²). All of these products are in `paper_products/reruns/`.
