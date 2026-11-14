# Data

Nothing in this release is observational data. Every data set is simulated by
the code in this repository, from seeds recorded in `config/heron.yaml`.

## Test data (needed for the validations)

Archive: `heron_testdata_v1.0.tar` (0.5 GB). `heron download testdata` fetches,
verifies and unpacks it in the repository root, where it creates `data/`
(`heron validate` does this by itself). Another location can be given with
`HERON_DATA`.

| path | content | used by |
|---|---|---|
| `data/mbhb/test/` | 2000 heterodyned MBHB test events of the `wide` set (signal only; noise is added with a fixed seed), `metadata.json` with the injected parameters | `evaluation/countdown.py`; default of every script with a `--data` option |
| `data/mbhb/dataset_info.json`, `bin_counts.npy`, `bin_freqs.npy` | grid and kernel of the MBHB set | the same |
| `data/mbhb_ridge/` | the same four items for the test split of the `ridge` set | `evaluation/eval_skygnpe.py` and `eval_robust.py` with `--data data/mbhb_ridge` |
| `data/ucb/test/` | 20000 UCB test windows with injected parameters | `evaluation/eval_ucb_flow.py`, `fig_ucb_recovery.py` |
| `data/yolo/images/test`, `data/yolo/labels/test` | 500 test spectrograms with ground-truth boxes | `yolo/detection_metrics.py`, `yolo/detection_figures.py` |

The two MBHB sets are the two training sets of `docs/TRAINING_PROVENANCE.md`:
`ridge` draws the spins of the heterodyne reference around the true ones,
`wide` draws them from the prior. The population sky areas and the robustness
sweep of the paper (`skygnpe_eval.npz`, `robust_sweep.npz`) were computed on the
`ridge` test split with the released weights, the countdown on the `wide` test
split. `python -m heron.run --stage validate` uses the same assignment
(`validate.sky_testset` in `config/heron.yaml`).

The test splits are distributed because each was generated in the same random
stream as its training set (training, then validation, then test), so
reproducing a test split alone would require generating the full training set
before it.

The other validations draw their events on the fly with the GPU waveforms and
need no stored data: `evaluation/pp_marginalized.py`, `sangria_marginalized.py`,
`ucb_subtract_test.py`, `recon_sky_test.py`, `yolo/validate_fullband.py`,
`heron/pipeline.py`.

## Training data (regenerated, not distributed)

`heron train --stage data` (`python -m heron.run --config config/heron.yaml
--stage data`) regenerates the training sets under `data/train/`, next to the
released test splits and without touching them: 6500 spectrogram images (1.4 GB), 10^6 UCB windows (2.4 GB) and
two MBHB sets of 10^7 samples (about 850 GB each). Generation is deterministic:
the first samples of the MBHB and UCB sets are compared with stored copies in
`tests/test_gpu_regression.py`. Seeds and the one caveat about the earlier MBHB
set are in `docs/TRAINING_PROVENANCE.md`.

## Numerical products (in the repository)

`paper_products/` (32 MB) holds the numbers behind the published figures:

| file | content | figure |
|---|---|---|
| `pp_ranks.npz`, `pp_ranks_cal.npz`, `pp_ranks_recal.npz` | percentile ranks of the truth: 500-event test set, 3000-event calibration set, recalibrated test set | Fig. 3a, Extended Data Fig. 4 |
| `pp_ranks_skycov.npz` | the same 500 events at reference scale 0.45, with the sky credibility of the truth | sky coverage in the main text |
| `pp_ranks_k16.npz` | 500 events with 16 pooled references | Supplementary Note (reference pooling) |
| `robust_sweep.npz` | 90% sky area for 250 events at 0, 10, 20, 40 UCBs | Fig. 3 (robustness panel) |
| `skygnpe_eval.npz` | sky area per pose iteration for 500 test events | sky area by SNR |
| `countdown_data.npy` | population sky area against time to merger | Fig. 4 |
| `handoff/sangria_synergy_*.npz` | the stage 1 to 3 hand-off of the Case-1 event (three data sets) | input of stage 4 |
| `pointing/synergy_*.npz` | stage-4 chains: clean, foreground, subtracted (seeded) and subtracted (unseeded) | Fig. 2, Fig. 5a,b |
| `pointing_ensemble_canonical.npz` | per-chain summary of the 24 ensemble chains | Fig. 5c |
| `reference_chains/case{1,2,3}_ptmcmc.npz` | 60,000 thinned cold-chain samples of the exact MCMC posteriors of arXiv:2602.22074 | alert against exact inference |
| `is_refine_case1.npz` | importance-sampling test of the pooled alert posterior | Supplementary Note |
| `sangria_1month_snr.npz` | one-month SNR of the Sangria galactic binaries with SNR > 1 | Methods (foreground density), Fig. 1b |
| `reruns/` | products of the reruns made from the released weights for this release | `docs/REPRODUCE.md` |

A script reads a product from `results/products/` if you have regenerated it
and from `paper_products/` otherwise, and prints which one it used.

## Stage-4 ensemble chains (optional)

Archive: `heron_stage4_ensemble_v1.0.tar` (0.35 GB; `heron download ensemble`): the 24 chains of the
noise-realization ensemble (hyperbolic and noise-fit Gaussian likelihood, 4
realizations, 3 data sets, 300,000 samples each) and the 12 hand-off files they
were run on. Unpack it in the repository root; it creates
`paper_products/pointing_ensemble/` and
`paper_products/pointing_ensemble_handoff/`.
`evaluation/fig_pointing_ensemble.py` then recomputes the areas from the chains;
without the archive it draws Fig. 5c from the stored summary.

## External input

`evaluation/resolved_population_1month.py` reads the public LISA Data Challenge
file `LDC2_sangria_training_v2.h5` (3 GB, https://lisa-ldc.lal.in2p3.fr), which
is not redistributed here. Its output is stored as
`paper_products/sangria_1month_snr.npz`.
