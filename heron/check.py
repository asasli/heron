"""
Installation check: environment, GPU, waveform builds, weights, one alert.

    python -m heron.check            # ~1 min on one GPU
    python -m heron.check --quick    # imports + checksums only (no GPU inference)

Exits non-zero at the first failed item, with the command that fixes it.
"""
import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PINNED = {'torch': '2.8', 'zuko': '1.5', 'cupy': '11.4', 'ultralytics': '8.4', 'numpy': '1.25'}


def step(name):
    print(f"[check] {name} ...", end=' ', flush=True)


def ok(msg=''):
    print(f"ok {msg}")


def fail(msg):
    print(f"FAILED\n        {msg}")
    raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()

    step("python packages")
    notes = []
    for mod, want in PINNED.items():
        try:
            v = importlib.import_module(mod).__version__
        except Exception as e:
            fail(f"cannot import {mod} ({e}); create the environment from environment.yml")
        if not v.startswith(want):
            notes.append(f"{mod} {v} (published runs used {want}.x)")
    ok(('; note: ' + ', '.join(notes)) if notes else '')

    step("GPU")
    import torch
    if not torch.cuda.is_available():
        fail("PyTorch sees no CUDA device; stages 1-3 need one GPU")
    ok(f"({torch.cuda.get_device_name(0)}, "
       f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB)")

    step("GPU waveforms (BBHx, GBGPU)")
    try:
        from bbhx.waveformbuild import BBHWaveformFD
        from gbgpu.gbgpu import GBGPU
        BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=True), use_gpu=True)
        GBGPU(use_gpu=True)
    except Exception as e:
        fail(f"{type(e).__name__}: {e}\n        build them with: bash scripts/install_waveforms.sh")
    ok()

    step("model weights (checksums)")
    from heron.artifacts import verify, WEIGHTS
    if not verify(strict=False, verbose=False):
        verify(strict=False, verbose=True)
        fail(f"weights in {WEIGHTS} are missing or differ from SHA256SUMS; "
             f"run `heron download --force` (docs/WEIGHTS.md)")
    ok(f"({WEIGHTS})")

    if args.quick:
        print("[check] quick check passed")
        return

    step("waveform regression (MBHB + UCB against stored values)")
    import numpy as np
    from tests.reference_values import waveform_fingerprint, REFERENCE
    fp = waveform_fingerprint()
    for k, ref in REFERENCE.items():
        if not np.allclose(fp[k], ref, rtol=1e-6):
            fail(f"{k}: got {fp[k]}, expected {ref}. The installed BBHx/GBGPU differ from the "
                 f"pinned commits (scripts/install_waveforms.sh).")
    ok()

    step("stages 1-3 on the Sangria Case-1 event")
    import tempfile
    from heron.stages import run_stages_1_to_3
    with tempfile.TemporaryDirectory() as d:
        r = run_stages_1_to_3(out=d, tag='check', verbose=False)
    s = r['summary']
    if abs(s['mbhb_snr'] - 357.0) > 1.0 or s['n_ucb_subtracted'] != 30:
        fail(f"unexpected result: SNR {s['mbhb_snr']:.1f}, {s['n_ucb_subtracted']} UCBs subtracted")
    ok(f"(SNR {s['mbhb_snr']:.0f}, alert 90% sky {s['alert_sky90_deg2']:.0f} deg^2 in "
       f"{s['timing_s']['alert_s']:.1f} s, {s['n_ucb_subtracted']} UCBs subtracted)")
    print("[check] installation verified")


if __name__ == '__main__':
    main()
