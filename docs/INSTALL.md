# Installation

Two environments are used. Stages 1 to 3, every validation and all training run
in the `heron` environment. Stage 4 uses the HyperWave package, which needs a
newer Python and a different build of the MBHB waveform, and has its own
environment.

## Requirements

- Linux, one NVIDIA GPU. The alert flow needs about 6 GB for inference and
  24 GB for training at the published batch size. The published runs used an
  RTX A6000 (stages 1 to 3, training) and an A100 (stage 4).
- An NVIDIA driver that supports CUDA 12 (PyTorch wheels) and a CUDA 11.x
  toolkit that provides `nvcc` (needed once, to build the two waveform
  packages). `scripts/install_waveforms.sh` reads its location from `CUDAHOME`
  (default `/usr/local/cuda`).
- conda (or mamba), `gcc`/`g++`, `git`.

## The `heron` environment

```bash
conda env create -f environment.yml
conda activate heron
CUDAHOME=/usr/local/cuda-11.6 bash scripts/install_waveforms.sh
pip install --no-deps -e .       # the `heron` command; editable install from the checkout
heron download                   # the released weights (0.5 GB)
heron check
```

`pip install --no-deps -e .` adds the `heron` command and makes the packages
importable from any directory. The install must be editable, because the code
reads `weights/`, `data/`, `config/` and `paper_products/` from the checkout.
Without it every command is available as a module from the repository root
(`python -m heron ...`, or `python -m heron.pipeline`, `python -m heron.run`,
`python -m heron.check`).

`environment.yml` pins every package to the version of the environment that
produced the published results (Python 3.9.15, PyTorch 2.8.0, zuko 1.5.0,
CuPy 11.4.0, Ultralytics 8.4.51, NumPy 1.25.2, Eryn at commit 5289664).

`scripts/install_waveforms.sh` builds the two GPU waveform packages from source
at the commits used for the paper:

| package | commit | date | role |
|---|---|---|---|
| [BBHx](https://github.com/mikekatz04/BBHx) | `a5aa738f381b865d317ba925efc9ca4ebf617a6c` | 2022-11-29 | MBHB waveform (PhenomD, (2,2) mode) and fast frequency-domain LISA response |
| [GBGPU](https://github.com/mikekatz04/GBGPU) | `10ad9ba21850add579084b31807e639106aadbf2` | 2023-09-20 | galactic-binary waveform |

The pinned BBHx commit matters: later BBHx releases changed the orbit and
response conventions and the Python interface, and the released weights were
trained on waveforms of this commit. The GBGPU commit is on a development
branch that also contains a shared-memory extension requiring NVIDIA cuFFTDx;
HERON uses only the standard waveform, so the script skips that extension and
points the build at the GSL of the conda environment. The nvcc flags of that
commit target compute capability 7.0 and 8.0 (V100, T4, A100, RTX 30/40 series,
RTX A-series).

`heron check` verifies the package versions, the GPU, both waveform
builds against stored waveform values (`tests/reference_values.py`), the
checksums of the weights, and runs stages 1 to 3 on the Sangria Case-1 event.

This procedure was tested on a clean environment created from these two files
on Ubuntu 20.04 with gcc 9.4, CUDA toolkit 11.6 and driver 535: the waveform
values are identical to those of the original environment, the stage 1 to 3
hand-off agrees with the published one to 3e-6 (single-precision flow sampling),
and the test suite passes.

A harmless warning appears if both a conda and a pip CuPy are present in the
same environment ("multiple CuPy packages are installed"); the environment file
installs only the conda one.

## Model weights and test data

```bash
heron download                # weights (needed for inference)
heron download testdata       # test splits (needed for `heron validate`)
heron download ensemble       # stage-4 ensemble chains (optional)
heron download --from DIR     # the same from archive files already on disk
```

The archives are checked against their SHA-256 and unpacked in the repository
root (`docs/WEIGHTS.md`, `docs/DATA.md`). `heron infer`, `heron check` and
`heron validate` fetch what they need if it is absent; set `HERON_NO_DOWNLOAD=1`
to forbid that. The locations can be changed with the environment variables
`HERON_WEIGHTS`, `HERON_DATA` and `HERON_RESULTS`, and the address of the
archives with `HERON_RELEASE_URL`.

## Stage 4: the pointing environment

Stage 4 (`pointing/synergy_pe.py`, `pointing/verify_conventions.py`) imports the
public HyperWave package, which provides the MBHB template in its own sampling
convention, the priors and the canonical sky-area estimator:

```bash
conda env create -f environment-pointing.yml
conda activate heron-pointing
python pointing/synergy_pe.py --npz paper_products/handoff/sangria_synergy_subtracted.npz \
    --force-backend cuda12x --smoke          # short chain: a functional test
```

The pipeline calls stage 4 automatically when it can find that interpreter:

```bash
export HERON_HYPERWAVE_PYTHON=$(conda run -n heron-pointing which python)
python -m heron.pipeline --simulate --case 1 --n-ucb 30 --out runs/case1
```

Without it the pipeline stops after writing the hand-off files and prints the
stage-4 command.

Status of stage 4 in this release. The published stage-4 chains were produced
on a cluster (one A100 per run). The environment file was tested as far as
creating the environment and importing HyperWave and its MBHB module. The
packaged BBHx wheel (1.2.3) could not be executed on the workstation used to
prepare the release, an AMD Ryzen 9 5950X, where importing it stops with an
illegal-instruction error, so the stage-4 sampler was not rerun for this
release. Stage 4 is therefore verified from its stored chains
(`paper_products/pointing/`, `tests/test_products.py`) and its code is provided
as run on the cluster. The directory of this repository that holds the stage-4
scripts is called `pointing/` so that it does not shadow the installed
`hyperwave` package.

## Running the tests

```bash
pytest tests/test_products.py tests/test_core.py     # CPU, seconds
pytest tests                                          # adds the GPU regression tests, about 1 min
```
