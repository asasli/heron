"""
HERON end-to-end pipeline on one month of (simulated) LISA data.

    [1] DETECT    spectrogram detector: resolved-UCB windows + MBHB flag
    [2] ALERT     amortized MBHB posterior on the UNSUBTRACTED data, conditioned
                  on the detected UCB count; post-hoc calibrated
    [3] CATALOG   per detected UCB: window flow -> F-statistic subtraction
                  (unresolved confusion stays in the data and in the PSD)
    [4] POINTING  alert-seeded exact inference on the subtracted data with the
                  public HyperWave package (pointing/synergy_pe.py)

Everything runs from the released weights; no training is needed.

    # the Sangria Case-1 event of the paper (stages 1-3: under a minute on one GPU)
    python -m heron.pipeline --simulate --case 1 --n-ucb 30 --out runs/case1

    # same data, stage 1 performed by the trained detector instead of injected windows
    python -m heron.pipeline --simulate --case 1 --n-ucb 30 --detector yolo --out runs/case1_yolo

    # another noise realization; stop before stage 4
    python -m heron.pipeline --simulate --seed 3 --skip-pointing --out runs/seed3

Outputs in --out:
    summary.json                 every number the run produced (areas, counts, timings)
    alert_posterior.npz          stage-2 posterior: raw and calibrated samples, 90% sky area
    ucb_catalogue.json           stage-3 catalogue (frequency, sky, fitted SNR per source)
    <tag>_{clean,foreground,subtracted}.npz + <tag>_README.txt   stage-4 hand-off
    pointing/                    stage-4 products (when HyperWave is installed)

Stage 4 needs the HyperWave package, which lives in its own environment
(docs/INSTALL.md). Without it the pipeline stops after the hand-off and prints
the command to run there. `--simulate` is the only data mode: the alert is
heterodyned against a reference that a search stage would provide, and in this
release that reference is drawn around the injection (Supplementary Note
"Stage-2 amortized inference"); see docs/SCOPE.md for what is and is not
demonstrated end to end.
"""
import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from heron.artifacts import require   # noqa: E402


def hyperwave_python():
    """Interpreter that can import the HyperWave package: $HERON_HYPERWAVE_PYTHON
    if set, else the current one when the package is installed in it."""
    exe = os.environ.get('HERON_HYPERWAVE_PYTHON')
    if exe:
        return exe
    spec = importlib.util.find_spec('hyperwave')
    # the installed package, not a stray directory on sys.path
    if spec is not None and spec.origin and 'site-packages' in spec.origin:
        return sys.executable
    return None


def run_pointing(subtracted_npz, out_dir, seed_mode='alert', backend=None, smoke=False):
    """Stage 4: alert-seeded parallel-tempered MCMC on the subtracted data
    (pointing/synergy_pe.py). Returns the output directory, or None when the
    HyperWave package is not available."""
    exe = hyperwave_python()
    out_dir = Path(out_dir).expanduser().resolve()
    cmd = [str(ROOT / 'pointing/synergy_pe.py'), '--npz',
           str(Path(subtracted_npz).expanduser().resolve()),
           '--seed-mode', seed_mode, '--out-dir', str(out_dir)]
    if backend:
        cmd += ['--force-backend', backend]
    if smoke:
        cmd += ['--smoke']
    if exe is None:
        print("[pointing] HyperWave is not installed in this environment; stage 4 skipped.")
        print("[pointing] run it in the HyperWave environment (docs/INSTALL.md):")
        print(f"[pointing]   python {shlex.join(cmd)}")
        print("[pointing] or set HERON_HYPERWAVE_PYTHON to that environment's interpreter.")
        return None
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"[pointing] {exe} {shlex.join(cmd)}")
    subprocess.run([exe] + cmd, cwd=ROOT, check=True)
    return out_dir


def run_event(case=1, n_ucb=30, t_pre=24.0, seed=0, detector='oracle', ref_scale=0.45,
              device='cuda:0', out='runs/demo', tag='event', verbose=True):
    """Stages 1 to 3 on one simulated event, from the weights in the weights
    directory. Writes the alert posterior, the UCB catalogue and the hand-off
    files into `out` and returns the result dictionary of
    heron.stages.run_stages_1_to_3 with the calibrated alert added
    (`alert_calibrated`, an [N, 11] array in the order ALERT_PARAM_NAMES)."""
    require('mbhb_flow', 'mbhb_norm', 'ucb_flow', *(('yolo',) if detector == 'yolo' else ()))
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    from heron.stages import (run_stages_1_to_3, recalibrate_alert, ALERT_PARAM_NAMES)
    from pe_smbhb.train_flow import sky_area_90
    r = run_stages_1_to_3(case=case, n_ucb=n_ucb, t_pre=t_pre, seed=seed,
                          device=device, detector=detector, ref_scale=ref_scale,
                          out=out, tag=tag, verbose=verbose)
    summary = r['summary']

    # stage-2 product: raw + calibrated alert posterior
    r['alert_calibrated'] = None
    r['alert_param_names'] = list(ALERT_PARAM_NAMES)
    if r['alert'] is not None:
        raw = r['alert']['alert_samples']
        cal = recalibrate_alert(raw).astype(np.float32)
        a90_cal = float(sky_area_90(cal[:4000, 7], cal[:4000, 6]))
        inj = r['event']['injection']
        truth = np.array([inj['Mc'], inj['q'], inj['chi1'], inj['chi2'], inj['logD'],
                          inj['cos_iota'], inj['sinb'], inj['lam'], inj['psi'], inj['phi0'],
                          inj['t_pre_hr']])
        np.savez(out / 'alert_posterior.npz', samples=cal, samples_raw=raw,
                 param_names=np.array(ALERT_PARAM_NAMES), truth=truth,
                 sky90_deg2=a90_cal, sky90_raw_deg2=r['alert']['_sky90'])
        summary['alert_sky90_calibrated_deg2'] = a90_cal
        q = np.percentile(cal, [5, 50, 95], axis=0)
        summary['alert_posterior'] = {n: dict(q05=float(q[0, i]), median=float(q[1, i]),
                                              q95=float(q[2, i]), injected=float(truth[i]))
                                      for i, n in enumerate(ALERT_PARAM_NAMES)}
        r['alert_calibrated'] = cal
        r['truth'] = truth
    (out / 'ucb_catalogue.json').write_text(json.dumps(
        dict(n_sources=len(r['catalogue']), sources=r['catalogue'],
             injected=r['event']['ucbs'], detection=r['detection']), indent=1, default=float))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=float))
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description="HERON end-to-end pipeline",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument('--simulate', action='store_true', help='simulated end-to-end run')
    ap.add_argument('--case', type=int, choices=(1, 2, 3), default=1, help='Sangria source (heron/sources.py)')
    ap.add_argument('--n-ucb', type=int, default=30, help='resolved UCBs injected')
    ap.add_argument('--t-pre', type=float, default=24.0,
                    help='hours between the end of the data and the merger')
    ap.add_argument('--seed', type=int, default=0, help='noise-realization seed')
    ap.add_argument('--detector', choices=['oracle', 'yolo'], default='oracle',
                    help='stage 1: injected windows with the measured centre error '
                         '(paper configuration) or the trained detector on the data')
    ap.add_argument('--ref-scale', type=float, default=0.45,
                    help='kernel scale of the heterodyne reference draw (published hand-off: 0.45)')
    ap.add_argument('--out', default='runs/demo')
    ap.add_argument('--tag', default='event')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--skip-pointing', action='store_true', help='stop after the hand-off')
    ap.add_argument('--seed-mode', choices=['alert', 'none'], default='alert',
                    help='stage-4 seeding (none = unseeded control)')
    ap.add_argument('--hw-backend', default=None, help='HyperWave bbhx backend, e.g. cuda12x')
    ap.add_argument('--pointing-smoke', action='store_true',
                    help='stage 4 with a short chain (a functional test, not a result)')
    args = ap.parse_args(argv)
    if not args.simulate:
        ap.error("--simulate is required (see docs/SCOPE.md for the data modes)")
    from heron.download import ensure_weights
    ensure_weights()
    out = Path(args.out)
    t00 = time.time()
    r = run_event(case=args.case, n_ucb=args.n_ucb, t_pre=args.t_pre, seed=args.seed,
                  detector=args.detector, ref_scale=args.ref_scale, device=args.device,
                  out=out, tag=args.tag)
    summary = r['summary']

    # stage 4
    if not args.skip_pointing:
        t0 = time.time()
        try:
            pdir = run_pointing(r['files']['subtracted'], out / 'pointing',
                                seed_mode=args.seed_mode, backend=args.hw_backend,
                                smoke=args.pointing_smoke)
        except (subprocess.CalledProcessError, OSError) as exc:
            summary['pointing'] = dict(ran=False, status='failed', error=str(exc),
                                       seconds=time.time() - t0)
            summary['total_wall_s'] = time.time() - t00
            (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=float))
            raise SystemExit(f'Stage 4 failed: {exc}') from exc
        summary['pointing'] = dict(ran=pdir is not None,
                                   status='completed' if pdir else 'unavailable',
                                   seconds=time.time() - t0, out=str(pdir) if pdir else None)
    else:
        summary['pointing'] = dict(ran=False, status='skipped', seconds=0, out=None)
    summary['total_wall_s'] = time.time() - t00
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=float))

    t = summary['timing_s']
    print("\n" + "=" * 68)
    print(f"HERON  case {summary['case']}  seed {summary['seed']}  MBHB SNR {summary['mbhb_snr']:.0f}")
    print(f"  [1] detect   ({summary['detector']}): {summary['n_ucb_detected']} UCB windows, "
          f"MBHB flag {summary['smbhb_flag']}   {t['detect_s']:.1f} s")
    if r['alert'] is not None:
        print(f"  [2] alert    {summary['alert_n_samples']} samples, 90% sky "
              f"{summary['alert_sky90_calibrated_deg2']:.0f} deg^2 (calibrated)   {t['alert_s']:.1f} s")
    print(f"  [3] catalog  {summary['n_ucb_subtracted']} UCBs subtracted, residual alpha "
          f"{summary['global_alpha']:.3f} (incl. confusion)   {t['catalogue_subtract_s']:.1f} s")
    p = summary.get('pointing')
    print(f"  [4] pointing {'done -> ' + p['out'] if p and p['ran'] else 'not run (hand-off written)'}")
    print(f"  summary: {out / 'summary.json'}    total wall {summary['total_wall_s']:.0f} s")
    print("=" * 68)


if __name__ == '__main__':
    main()
