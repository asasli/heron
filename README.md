# HERON: LISA Global Fit with AI

Code, model weights and data products for the paper *LISA as a continuous
early-warning instrument for massive black-hole binary mergers* (Sasli et al.).

HERON takes one month of simulated LISA TDI data and returns a sky localization
of a massive black-hole binary (MBHB) before it merges, in four stages:

| stage | what it does | code |
|---|---|---|
| 1 Detect | spectrogram detector: resolved galactic binaries (UCBs) and the MBHB flag | `yolo/` |
| 2 Alert | amortized 11-parameter MBHB posterior on the unsubtracted data, conditioned on the detected UCB count | `pe_smbhb/` |
| 3 Catalogue | per-UCB window flow, then F-statistic subtraction of every resolved UCB | `pe_ucb/` |
| 4 Point | alert-seeded exact inference on the subtracted data with the HyperWave package | `pointing/` |

`heron/` holds the orchestration and the `heron` command. There are two ways to
use the package, and they share all the code:

- **Inference only.** The released weights are fetched once (0.5 GB) and the
  pipeline runs on them. Nothing is trained.
- **Training from scratch.** One command generates the three training sets,
  trains the three networks, fits the calibration map, writes a weights bundle
  with the same file names as the released one, and runs the pipeline and the
  validations on it.

## Install

```bash
conda env create -f environment.yml        # pinned versions (docs/INSTALL.md)
conda activate heron
bash scripts/install_waveforms.sh          # BBHx and GBGPU at the pinned commits
pip install --no-deps -e .                 # the `heron` command
```

## Inference from the released weights

```bash
heron download                             # weights, verified against their checksums
heron check                                # packages, GPU, waveforms, weights, one event
heron infer --case 1 --n-ucb 30 --out runs/case1
```

`heron infer` downloads the weights by itself if they are absent, so the last
line alone is enough. It prints the stage summary and writes
`runs/case1/summary.json`, the alert posterior, the UCB catalogue and the three
hand-off files for stage 4. With the default seed the hand-off files reproduce
the published ones to floating-point rounding; `tests/test_gpu_regression.py`
asserts this.

```
HERON  case 1  seed 0  MBHB SNR 357
  [1] detect   (oracle): 30 UCB windows, MBHB flag True
  [2] alert    32000 samples, 90% sky 17164 deg^2 (calibrated)
  [3] catalog  30 UCBs subtracted, residual alpha 0.505 (incl. confusion)
  [4] pointing not run (hand-off written)
```

Variants:

```bash
heron infer --detector yolo --out runs/case1_yolo   # stage 1 by the trained detector on the same data
heron infer --seed 3 --out runs/seed3               # another noise realization
heron infer --case 2 --out runs/case2               # another Sangria binary
heron report --run runs/case1                       # interactive HTML dashboard of the run
```

The same from Python:

```python
from heron import Heron

h = Heron.from_pretrained()                  # released weights, fetched once if absent
r = h.run(case=1, n_ucb=30, seed=0, out="runs/case1")
r.summary                                    # counts, areas, timings
r.alert                                      # calibrated alert posterior, array [N, 11]
r.catalogue                                  # the UCB catalogue of stage 3
r.handoff                                    # the three files for stage 4
```

Stage 4 runs in a separate environment that contains the HyperWave package
(`docs/INSTALL.md`, "Stage 4"):

```bash
python pointing/synergy_pe.py --npz runs/case1/event_subtracted.npz --force-backend cuda12x
```

## Training from scratch

```bash
heron train --smoke            # every stage at toy scale: 6 to 12 minutes, 1 GPU
heron train                    # the published configuration: about 10 days, 1.8 TB
heron train --dry-run          # print every command without running it
```

The stages, in order: detector images, detector training, UCB windows, UCB
flow, UCB line response, MBHB training sets, alert flow (four phases),
calibration map, export of the weights bundle, pipeline on one event,
validations. A finished stage is skipped, so an interrupted run continues with
the stage (or training phase) it was in. Everything is defined in `config/heron.yaml`; the
seeds are in the file. The bundle is written to `<out>/weights`, and the
pipeline runs on it with

```bash
heron infer --weights runs/paper/weights --out runs/mine
```

(`Heron.from_pretrained("runs/paper/weights")` from Python). What a retraining
reproduces, and what it cannot, is described in `docs/TRAINING_PROVENANCE.md`.

## Three levels of reproduction

Everything in the paper can be checked at three depths. Each level is one
command, and each is independent of the ones below it.

| level | what you rerun | needs | time | command |
|---|---|---|---|---|
| 0 | every figure and quoted number, from the stored numerical products | CPU, no weights | minutes | `heron figures` and `pytest tests/test_products.py` |
| 1 | inference and every validation, from the released weights | 1 GPU, weights (0.5 GB), test data (0.5 GB) | about 4 h for the full set, 1 min for one event | `heron infer` and `heron validate` |
| 2 | data generation and training of the three networks from scratch | 1 GPU, 1.8 TB disk | about 10 days | `heron train` |
| 2 (toy scale) | the complete level-2 code path with tiny data sets and networks | 1 GPU | 6 to 12 minutes | `heron train --smoke` |

`docs/REPRODUCE.md` lists, for every figure, table and number of the paper, the
command that produces it, the stored product behind it, the published value and
the value obtained when this release was rerun from the weights.

## Measured run times

One NVIDIA RTX A6000 (48 GB), this release, Sangria Case-1 event:

| step | time |
|---|---|
| stage 1, tiled detector over 0.3 to 4 mHz (71 tiles) | 1.7 s |
| stage 2, one flow evaluation, 1,000 posterior samples | 0.32 s |
| stage 2, calibrated alert (3 pose iterations, 3,200 samples) | 3.2 s |
| stage 2, pipeline configuration (3 pose iterations, 32,000 samples) | 15.5 s |
| stage 3, 30 UCBs catalogued and subtracted | 5.8 s |
| stage 4, alert-seeded exact inference (A100, published runs) | 14 to 19 min |

The cost of stage 2 is the flow sampling, 0.38 ms per posterior sample beyond
the first thousand.

## What the release contains

```
heron/            cli.py (the `heron` command), api.py (Python interface),
                  pipeline.py (four stages on one event), run.py (data, training,
                  validation from YAML), stages.py (the stage functions),
                  download.py and artifacts.py (weights: fetch, paths, checksums),
                  check.py (installation check)
yolo/             stage 1: dataset generator, training, single-band and tiled
                  full-band inference, full-band validation, test-set metrics
pe_smbhb/         stage 2: heterodyned data generation, networks, training, inference
pe_ucb/           stage 3: UCB window data generation and flow training
pointing/         stage 4: hand-off export, alert-seeded sampler, conventions
                  check, ensemble harness
evaluation/       every validation and figure script of the paper
config/           heron.yaml (published configuration), smoke.yaml (toy scale)
paper_products/   the numerical products behind the published figures (32 MB)
tests/            regression tests against the published values
webpage/          interactive HTML dashboard & science portal reflecting the paper
docs/             INSTALL, REPRODUCE, TRAINING_PROVENANCE, WEIGHTS, DATA, SCOPE
weights/          model weights (`heron download`, 0.5 GB)
data/             held-out test splits (`heron download testdata`, 0.5 GB)
```

## Scope of what is demonstrated

All data are simulated (`docs/SCOPE.md`). Three points matter when reading the
results:

1. Stage 1 hand-off. The end-to-end numbers of the paper place a window on every
   injected UCB with the measured centre error of the detector (`--detector
   oracle`), and the detector itself is validated on separate spectrograms.
   `--detector yolo` runs the released detector on the same simulated data that
   stages 2 and 3 consume.
2. Heterodyne reference. The alert is heterodyned against a reference waveform
   that a search stage would supply. Here the reference is drawn around the
   injected source with the training kernel.
3. Stage 4. `pointing/synergy_pe.py` is the alert-seeded sampler with a Gaussian
   likelihood and fixed noise level that produced the single-realization areas
   (191, 134 and 143 deg²). The ensemble with the heavier-tailed likelihood
   (Fig. 5c) was run with scripts of the HyperWave repository on a cluster; its
   chains are distributed with this release and the figure is rebuilt from them.

## Documentation

- `docs/INSTALL.md`: environment, GPU waveform builds, stage-4 environment
- `docs/REPRODUCE.md`: figure-by-figure and number-by-number reproduction table
- `docs/TRAINING_PROVENANCE.md`: how the released weights were trained
- `docs/WEIGHTS.md`, `docs/DATA.md`: the archives and their checksums
- `docs/SCOPE.md`: what is simulated, what is assumed, what is outside this release
- each component directory has its own README

## Development and CPU checks

Use an editable checkout; configuration files and published products are read
from the repository. The CUDA runtime remains defined by `environment.yml`.
For CLI, orchestration and numerical-product tests without CUDA:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest tests -q
```

GPU and weight-dependent tests skip when their prerequisites are absent.
GitHub Actions runs this suite on Python 3.9 and 3.11. This does not replace
`heron train --smoke` and the GPU regressions in the scientific environment.

Explicit CLI paths, including `--config`, are relative to the caller's directory.
Paths inside training YAML files remain relative to the checkout. Both
`--out path` and `--out=path` accept spaces. Stage commands run without a shell;
their output is captured in `<out>/logs`.

Training reuses existing outputs unless `--force` is passed. An MBHB phase
requires its completion marker, checkpoint and normalization file before it can
be skipped. Use a new output directory or `--force` after changing training
settings; existing markers do not track configuration changes. Export refuses
a bundle missing any of the eight expected model artifacts.

Stage 4 records `completed`, `unavailable`, `skipped` or `failed` in
`summary.json`. A failed pointing process exits with an error and preserves the
stage 1–3 products for inspection or retry.

## Licence and citation

The code is released under the MIT licence (`LICENSE`). The weights and data
products are released under CC BY 4.0. Please cite the paper (`CITATION.cff`).
BBHx, GBGPU, Eryn, zuko and Ultralytics YOLO are separate packages under their
own licences.
