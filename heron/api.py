"""
Python interface to the pipeline.

    from heron import Heron

    h = Heron.from_pretrained()                 # the released weights (fetched once if absent)
    r = h.run(case=1, n_ucb=30, seed=0, out='runs/case1')
    r.summary                                   # counts, areas, timings
    r.alert                                     # calibrated alert posterior, [N, 11]
    r.alert_param_names                         # Mc, q, chi1, chi2, logD, cos_iota, sin_beta, lambda, psi, phi0, t_pre_hours
    r.catalogue                                 # stage-3 UCB catalogue
    r.handoff                                   # the three files for stage 4

    h = Heron.from_pretrained('runs/paper/weights')   # a bundle written by `heron train`

The weights directory is fixed when the first HERON module that needs it is
imported, so choose it first: create one `Heron` per Python process.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Result:
    """Outcome of stages 1 to 3 on one event."""

    def __init__(self, r, out):
        self.raw = r
        self.out = Path(out)
        self.summary = r['summary']
        self.alert = r['alert_calibrated']
        self.alert_raw = r['alert']['alert_samples'] if r['alert'] is not None else None
        self.alert_param_names = r['alert_param_names']
        self.truth = r.get('truth')
        self.catalogue = r['catalogue']
        self.detection = r['detection']
        self.handoff = {k: Path(v) for k, v in r['files'].items()}

    def __repr__(self):
        s = self.summary
        return (f"<heron Result case {s['case']} seed {s['seed']}: MBHB SNR {s['mbhb_snr']:.0f}, "
                f"{s['n_ucb_subtracted']} UCBs subtracted, alert 90% sky "
                f"{s.get('alert_sky90_calibrated_deg2', float('nan')):.0f} deg^2>")


class Heron:
    """The pipeline bound to one weights directory."""

    def __init__(self, weights, device='cuda:0'):
        self.weights = Path(weights)
        self.device = device

    @classmethod
    def from_pretrained(cls, weights=None, device='cuda:0', download=True):
        """Bind the pipeline to a weights directory.

        weights=None  the released weights in <repository>/weights (or
                      $HERON_WEIGHTS); downloaded once if absent and
                      `download` is true.
        weights=DIR   a bundle with the same file names, as written by
                      `heron train` in <run>/weights.
        """
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        if weights is not None:
            w = Path(weights).expanduser().resolve()
            if not w.is_dir():
                raise FileNotFoundError(f"weights directory {w} does not exist")
            loaded = sys.modules.get('heron.artifacts')
            if loaded is not None and Path(loaded.WEIGHTS).resolve() != w:
                raise RuntimeError(
                    f"this process already uses the weights in {loaded.WEIGHTS}; "
                    f"start a new process (or set HERON_WEIGHTS before importing heron) to use {w}")
            os.environ['HERON_WEIGHTS'] = str(w)
        elif download:
            from heron.download import ensure_weights
            ensure_weights()
        from heron import artifacts
        artifacts.require('mbhb_flow', 'mbhb_norm', 'ucb_flow')
        return cls(artifacts.WEIGHTS, device)

    def verify(self):
        """Check every file of the weights directory against its SHA256SUMS."""
        from heron import artifacts
        return artifacts.verify(strict=False)

    def run(self, case=1, n_ucb=30, t_pre=24.0, seed=0, detector='oracle', ref_scale=0.45,
            out='runs/demo', tag='event', verbose=True):
        """Stages 1 to 3 on one simulated event (heron/pipeline.py: run_event)."""
        from heron.pipeline import run_event
        r = run_event(case=case, n_ucb=n_ucb, t_pre=t_pre, seed=seed, detector=detector,
                      ref_scale=ref_scale, device=self.device, out=out, tag=tag, verbose=verbose)
        return Result(r, out)

    def __repr__(self):
        return f"<Heron weights={self.weights} device={self.device}>"
