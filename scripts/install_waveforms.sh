#!/usr/bin/env bash
# Build the two GPU waveform packages at the exact commits used for the paper:
#
#   BBHx   a5aa738f  (2022-11-29)  massive black-hole binaries, PhenomD/HM + fast FD response
#   GBGPU  10ad9ba2  (2023-09-20)  galactic binaries (development branch)
#
# Later BBHx releases changed the orbit/response conventions and the API, so
# the pinned commit is required to reproduce the published numbers
# (tests/test_waveform_regression.py checks the result against stored values).
#
# Run inside the activated `heron` conda environment:
#     bash scripts/install_waveforms.sh
# Needs: nvcc from a CUDA 11.x toolkit (CUDAHOME, default /usr/local/cuda),
#        gcc/g++, and the conda packages of environment.yml (gsl, lapack 3.6.1).
set -euo pipefail

: "${CONDA_PREFIX:?activate the heron conda environment first}"
export CUDAHOME="${CUDAHOME:-/usr/local/cuda}"
[ -x "$CUDAHOME/bin/nvcc" ] || { echo "nvcc not found in $CUDAHOME/bin (set CUDAHOME)"; exit 1; }
BUILD="${HERON_BUILD_DIR:-$(mktemp -d)}"
mkdir -p "$BUILD"
echo "building in $BUILD with CUDAHOME=$CUDAHOME, prefix $CONDA_PREFIX"
export CPATH="$CONDA_PREFIX/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LDFLAGS="-Wl,-rpath,$CONDA_PREFIX/lib ${LDFLAGS:-}"

# ---------------------------------------------------------------- BBHx
cd "$BUILD"
rm -rf BBHx
git clone -q https://github.com/mikekatz04/BBHx.git
cd BBHx
git checkout -q a5aa738f381b865d317ba925efc9ca4ebf617a6c
python setup.py install --gsl "$CONDA_PREFIX" --lapack "$CONDA_PREFIX"

# --------------------------------------------------------------- GBGPU
cd "$BUILD"
rm -rf GBGPU
git clone -q https://github.com/mikekatz04/GBGPU.git
cd GBGPU
git fetch -q origin 10ad9ba21850add579084b31807e639106aadbf2
git checkout -q 10ad9ba21850add579084b31807e639106aadbf2
# The development commit hard-codes the GSL location and builds an extra
# shared-memory extension that needs NVIDIA cuFFTDx. HERON uses only the
# standard waveform (gbgpu.gbgpu.GBGPU.run_wave), so point GSL at the conda
# prefix and skip that extension.
python - <<'PY'
import os, re
s = open('setup.py').read()
p = os.environ['CONDA_PREFIX']
s = s.replace('lib_gsl_dir = "/opt/local/lib"', f'lib_gsl_dir = "{p}/lib"')
s = s.replace('include_gsl_dir = "/opt/local/include"', f'include_gsl_dir = "{p}/include"')
s = s.replace(', "cufftdx/include"', '')
# drop the cuFFTDx-dependent extension from the list handed to setup()
s = re.sub(r'\bext_gpu2\s*,\s*', '', s)
s = re.sub(r',\s*ext_gpu2\b', '', s)
open('setup.py', 'w').write(s)
# gbgpu.py imports names from that extension inside its GPU block and stores
# them on the waveform object. Replace each import by a None placeholder, so
# the GPU waveform (run_wave) works as usual without the extension.
out = []
for line in open('gbgpu/gbgpu.py').read().splitlines(keepends=True):
    m = re.match(r'^(\s*)from gbgpu\.sharedmem import (.+)$', line.rstrip('\n'))
    if m:
        names = [n.strip() for n in m.group(2).split(',') if n.strip()]
        out.append(f"{m.group(1)}{' = '.join(names)} = None  # cuFFTDx extension not built\n")
    else:
        out.append(line)
open('gbgpu/gbgpu.py', 'w').writelines(out)
PY
python setup.py install

cd /
python - <<'PY'
from bbhx.waveformbuild import BBHWaveformFD
from gbgpu.gbgpu import GBGPU
BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=True), use_gpu=True)
GBGPU(use_gpu=True)
print("BBHx and GBGPU GPU builds import correctly")
PY
