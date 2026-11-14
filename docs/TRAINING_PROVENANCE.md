# How the released weights were trained

This file records what was actually run to produce the weights in `weights/`.
`config/heron.yaml` encodes the same sequence for `python -m heron.run`. The
original run scripts and training logs are kept in `docs/provenance/`.

## Summary

| artifact | trained from | data | schedule | wall time (RTX A6000) |
|---|---|---|---|---|
| `yolo_lisa_best.pt` | `yolo26n_base.pt` (pretrained YOLO nano) | 5000 / 1000 / 500 spectrogram images, seed 42 | 80 epochs, batch 16, patience 20, seed 0, deterministic | 1.5 h |
| `ucb_flow.pt` | scratch | 10^6 UCB windows, seed 42 | 40 epochs, batch 1024, lr 3e-4, cosine | 35 min |
| `ucb_response.npz` | `evaluation/calibrate_ucb.py` | 120 frequencies, 8 orientations each, seed 0 | no training | 20 s |
| `mbhb_flow.pt`, `mbhb_norm.pt` | scratch, four phases | two sets of 10^7 heterodyned MBHB samples | see below | about 8.5 days |
| `recal_map.npz` | `evaluation/fit_recalibration.py` | 3000 injections through the trained flow | no training | 3 h |

## Detector (stage 1)

Run `lisa_20260323_031307`. All training arguments are in
`docs/provenance/yolo/args.yaml` and the per-epoch metrics in
`docs/provenance/yolo/results.csv`. The released file is the best-validation
checkpoint of that run (identical checksum). The base checkpoint is released as
`weights/yolo26n_base.pt` so that the fine-tuning starts from the same weights.

Dataset: 5000 training images with 1 to 5 UCBs and a 50% MBHB probability, 1000
validation and 500 test images with 10 to 20 UCBs, all sources above SNR 10,
one 110 µHz band (2.53 to 2.64 mHz), `yolo/yolo_config.py`.

## UCB window flow (stage 3)

`python -m pe_ucb.data_ucb --n-train 1000000 --n-val 20000 --n-test 20000 --seed 42`
followed by `python -m pe_ucb.train_flow --epochs 40`. Log:
`docs/provenance/ucb/train.log` (best validation NLL -7.589 at epoch 40).
Regenerating the data with seed 42 reproduces the stored windows bit for bit
(`tests/test_gpu_regression.py`).

## Alert flow (stage 2)

Architecture: residual 1-D CNN encoder with FiLM conditioning (hidden 384, 12
blocks, context 768), matched-filter bank of 256 kernels, neural spline flow
with 40 transforms (hidden 768, 16 bins); 108,775,441 parameters. Optimizer
AdamW, weight decay 1e-3, batch 256, auxiliary sky-flow weight 0.5, 10-epoch
warm-up then cosine decay, training set rotated over 200k-sample chunks every 4
epochs, instrument noise and the UCB foreground resampled every batch.

The released model is the result of four consecutive runs. Each run resumes
from the best-sky checkpoint of the previous one.

| phase | config tag | original tag | data set | epochs | learning rate | targets | note |
|---|---|---|---|---|---|---|---|
| 1 | `p1_scratch` | `het_v13scale` | ridge | 152 of a 280-epoch schedule | 1.5e-4 | residual (Mc, q, spins, t) | from scratch; interrupted by a machine reboot at epoch 152 |
| 2 | `p2_anneal` | `het_v13scale_r2` | ridge | 130 | 1.0e-4 | residual | completes the anneal; best-sky checkpoint at epoch 109 |
| 3 | `p3_abs` | `het_v14abs` | ridge | 110 | 6.0e-5 | Mc, t residual; q and spins absolute | best-sky checkpoint at epoch 110 |
| 4 | `p4_widespin` | `het_v15wspin` | wide | 130 | 6.0e-5 | as phase 3 | released model: best-sky checkpoint at epoch 128 |

About 23 minutes per 200k-sample epoch; 59, 52, 43 and 50 hours for the four
phases.

Why four phases. Phases 1 and 2 are one training run split by the reboot.
Phase 3 changed the mass ratio and the spins from kernel-scaled residuals to
absolute targets, because the residual encoding limited their posterior width
to the width of the reference kernel. Phase 4 replaced the training set by one
whose reference spins are drawn from the prior, so that the network measures
the spins from the data. A single run from scratch on the `wide` set with
absolute targets is expected to reach an equivalent model; it has not been
run, and the released weights are the four-phase model.

### The two MBHB training sets

Both contain a base set (200k training, 5k validation, 2k test samples) and 49
chunks of 200k training samples, about 850 GB each
(`pe_smbhb/data_mbhb.py`).

- `wide` (phase 4, `--wide-spin-ref`): base seed 90000, chunk *i* seed 90000 + *i*,
  *i* = 1 to 49. This is exactly the published set: regenerating with these
  seeds reproduces the stored samples bit for bit
  (`tests/test_gpu_regression.py`). Its `dataset_info.json` is released as
  `weights/mbhb_dataset_info.json`, and its test split is the one distributed
  for the validations.
- `ridge` (phases 1 to 3, truth-centred reference spins): chunks 48 to 96 of an
  earlier campaign. Chunks 64 to 96 used seed 70000 + *i*. The seeds of the base
  set and of chunks 48 to 63 were not logged. `config/heron.yaml` uses
  70000 + *i* throughout, which gives statistically equivalent, not identical,
  samples for those chunks.

### Logs

`docs/provenance/mbhb/phase{2,3,4}_*.log` are the original training logs
(progress bars removed, absolute paths shortened). The log of phase 1 was lost
in the reboot that interrupted it; its checkpoint at epoch 152 is the starting
point recorded in the phase-2 log. `original_run_*.sh` are the scripts that
launched the four phases and the two data campaigns.

## Calibration map

`recal_map.npz` holds, for each parameter, the sorted percentile ranks of the
truth in 3000 independent injections (`paper_products/pp_ranks_cal.npz`):
the empirical cumulative distribution used by the post-hoc calibration.
Refitting from that file reproduces the released map exactly
(`tests/test_products.py`). The calibration and test sets were produced at
reference kernel scale 1.0 (`docs/REPRODUCE.md`, "Reference scale").

## What a retraining reproduces

Data generation is deterministic given the seeds. Network training is seeded
(`--seed`), but GPU convolution and scatter operations are not bit-deterministic,
and the published runs were made before the training seed was fixed. A
retraining therefore gives a model that is statistically equivalent to the
released one, with different weights. The checks that such a model has to pass
are the validations of `python -m heron.run --stage validate`.

`python -m heron.run --config config/smoke.yaml --stage all` runs the identical
sequence (three data sets, three trainings in the same order, four alert-flow
phases with resume, calibration map, export, pipeline, validations) with toy
sizes in about six minutes, which verifies the code path.
