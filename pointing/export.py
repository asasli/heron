"""
Write the stage-4 hand-off for one simulated event (stages 1-3 of the pipeline).

Produces three npz files for the same MBHB and the same noise realization:

  <tag>_clean.npz       MBHB + instrument noise
  <tag>_foreground.npz  MBHB + resolved UCBs + confusion + instrument noise
  <tag>_subtracted.npz  the foreground data after HERON removal of the resolved UCBs

npz keys:
  freqs [Hz], df, Tobs, dt     frequency grid (0.1-4 mHz) and scalars
  dA, dE  [complex]            TDI A/E data (raw, not whitened)
  psdA, psdE                   recommended one-sided PSD for that file
  psd_instr, psd_conf          instrument and confusion PSD separately
  hA_inj, hE_inj               the noiseless injected MBHB template
  injection, conventions       injected parameters and cross-code conventions
  alert_samples [N, 11]        stage-2 alert posterior (columns alert_param_names);
                               for chain initialization and Metropolis-Hastings
                               proposals in stage 4, NOT a prior (same data)
  track_t, track_f_{lo,med,hi} Newtonian chirp-track quantiles of the alert
  n_ucb_injected, n_ucb_subtracted, global_alpha

With the default arguments this reproduces the published hand-off of the
Sangria Case-1 binary to floating-point rounding
(tests/test_handoff_regression.py).

    python pointing/export.py --case 1 --n-ucb 30 --tag sangria_synergy --out exports
    python pointing/export.py --seed 2 ...        # another noise realization (ensemble)
    python pointing/export.py --detector yolo ... # run the trained detector on the data
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from heron.artifacts import MBHB_FLOW, MBHB_NORM, UCB_FLOW, require   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--case', type=int, default=1, help='Sangria source (heron/sources.py)')
    ap.add_argument('--n-ucb', type=int, default=30, help='resolved UCBs injected')
    ap.add_argument('--t-pre', type=float, default=24.0,
                    help='hours between the end of the data and the merger')
    ap.add_argument('--detector', choices=['oracle', 'yolo'], default='oracle',
                    help='stage 1: injected windows with the measured centre error '
                         '(paper configuration) or the trained detector on the data')
    ap.add_argument('--model', default=UCB_FLOW, help='UCB window flow')
    ap.add_argument('--mbhb-model', default=MBHB_FLOW)
    ap.add_argument('--mbhb-norm', default=MBHB_NORM)
    ap.add_argument('--data', default=None,
                    help='mbhb dataset root of a retrained model (reference kernel); '
                         'default: the kernel released with the weights')
    ap.add_argument('--ref-scale', type=float, default=0.45)
    ap.add_argument('--no-alert', action='store_true',
                    help='skip the stage-2 alert posterior')
    ap.add_argument('--tag', default='sangria_synergy')
    ap.add_argument('--out', default='exports')
    ap.add_argument('--seed', type=int, default=0,
                    help='noise-realization seed (vary for an ensemble)')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    require('ucb_flow', *(() if args.no_alert else ('mbhb_flow', 'mbhb_norm')),
            *(('yolo',) if args.detector == 'yolo' else ()))

    from heron.stages import run_stages_1_to_3
    run_stages_1_to_3(case=args.case, n_ucb=args.n_ucb, t_pre=args.t_pre, seed=args.seed,
                      device=args.device, detector=args.detector, no_alert=args.no_alert,
                      ref_scale=args.ref_scale, model=args.mbhb_model, norm=args.mbhb_norm,
                      ucb_flow=args.model, data=args.data, out=args.out, tag=args.tag)
    print("done.")


if __name__ == '__main__':
    main()
