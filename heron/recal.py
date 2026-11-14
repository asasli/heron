"""
Post-hoc calibration of the alert posterior (Methods, "Post-hoc calibration").

A per-parameter monotonic map g_p, the empirical cumulative distribution of the
percentile ranks of the truth over an independent calibration set
(evaluation/fit_recalibration.py), corrects the small reference-echo offset of
the intrinsic parameters. It is learned once, released as
weights/recal_map.npz and applied to every alert posterior. Each sample keeps
its marginal rank order, so the correlations between parameters are unchanged.

numpy and scipy only: the figure scripts use it without the GPU stack.
"""
import os

import numpy as np


def load_recal(path=None):
    """The calibration map as {parameter: sorted calibration ranks}, or None if
    the file is absent. Default: the released map; if the weights are not
    installed, the same map rebuilt from the published calibration-set ranks."""
    from heron.artifacts import MBHB_RECAL, PAPER_PRODUCTS
    path = path or MBHB_RECAL
    if os.path.exists(path):
        z = np.load(path)
        return {str(p): z[f'g_{p}'] for p in z['params']}
    cal = PAPER_PRODUCTS / 'pp_ranks_cal.npz'
    if str(path) == MBHB_RECAL and cal.exists():
        z = np.load(cal)
        return {k: np.sort(z[k]) for k in z.files if k not in ('N', 'n_ucb', 'sky_cred')}
    return None


def recalibrate_samples(samples, recal, param_order):
    """Calibrated posterior samples: x' = F^{-1}(g_p^{-1}(u)) with u the
    marginal rank of each sample (order-preserving, so the joint structure is
    preserved). Columns without a map are returned unchanged."""
    if recal is None:
        return samples
    from scipy.stats import rankdata
    out = np.array(samples, dtype=float, copy=True)
    n = len(samples)
    grid = (np.arange(n) + 0.5) / n
    for j, p in enumerate(param_order):
        g = recal.get(p)
        if g is None:
            continue
        gs = np.sort(np.asarray(g, float)); m = len(gs)
        x = samples[:, j]; sx = np.sort(x)
        u = (rankdata(x, method='average') - 0.5) / n        # flow CDF of each sample
        ginv = np.interp(u, (np.arange(m) + 0.5) / m, gs)    # g_p^{-1}(u)
        out[:, j] = np.interp(ginv, grid, sx)                # flow quantile at g^{-1}(u)
    return out
