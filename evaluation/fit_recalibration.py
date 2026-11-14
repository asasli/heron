"""Post-hoc calibration of the amortized MBHB posterior.

The flow carries a small (<=0.15 sigma) learned offset on the intrinsic
parameters (the reference-echo floor); the extrinsics are unbiased. Here we
learn a per-parameter monotonic recalibration map g_p (the empirical rank CDF)
on an INDEPENDENT calibration set and validate it on a held-out test set: if
the recalibrated test-set ranks are uniform, the miscalibration is stationary
and the correction generalizes (it is not fit to the test data).

  fit:      g_p = empirical CDF of the calibration-set ranks
  apply:    r -> g_p(r)     (monotone; identity where already calibrated)
  validate: KS(recalibrated test ranks, uniform)

Inputs are the percentile-rank files written by evaluation/pp_marginalized.py
(the published ones in paper_products/ unless you have rerun them):
  pp_ranks_cal.npz   calibration set, 3000 injections
  pp_ranks.npz       held-out test set, 500 injections
Outputs (results/): recal_map.npz (the maps), pp_ranks_recal.npz (recalibrated
test ranks) and fig_pp.pdf (Fig. 3a). The released map is weights/recal_map.npz;
copy a refitted map there to use it in the pipeline.

Usage:
    python evaluation/fit_recalibration.py [--cal <npz>] [--test <npz>]
"""
import argparse
import sys
import numpy as np
from pathlib import Path
from scipy.stats import kstest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from heron.artifacts import (MBHB_FLOW, MBHB_NORM, MBHB_RECAL, UCB_FLOW, UCB_RESPONSE,
                             MBHB_DATA, UCB_DATA, load_kernel, product_out,
                             product_in, figure_out)

PARAMS = ['Mc', 'q', 'c1', 'c2', 'chi_eff', 'logD', 'cos_inc', 'sin_beta',
          'lambda', 'psi', 'phi0', 't_pre_hours']


def fit_map(cal_ranks):
    """Empirical rank-CDF (sorted cal ranks). apply = searchsorted/len."""
    return np.sort(np.asarray(cal_ranks, float))


def apply_map(g, ranks):
    return np.searchsorted(g, np.asarray(ranks, float), side='right') / len(g)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cal', default=None, help='calibration-set ranks (default: pp_ranks_cal.npz product)')
    ap.add_argument('--test', default=None, help='test-set ranks (default: pp_ranks.npz product)')
    ap.add_argument('--out-map', default=None, help='default results/products/recal_map.npz')
    args = ap.parse_args()
    args.cal = args.cal or product_in('pp_ranks_cal.npz')
    args.test = args.test or product_in('pp_ranks.npz')
    args.out_map = args.out_map or product_out('recal_map.npz')

    cal = np.load(args.cal); test = np.load(args.test)
    maps = {}
    print(f"cal N={int(cal['N'])}  test N={int(test['N'])}")
    print(f"{'param':12s} {'KS_before':>10s} {'KS_after':>10s} {'bias_b':>8s} {'bias_a':>8s}")
    rows = {}
    for p in PARAMS:
        if p not in cal.files or p not in test.files:
            continue
        g = fit_map(cal[p]); maps[p] = g
        rb = np.asarray(test[p]); ra = apply_map(g, rb)
        ksb = kstest(rb, 'uniform').pvalue; ksa = kstest(ra, 'uniform').pvalue
        bb = np.mean(rb) - 0.5; ba = np.mean(ra) - 0.5
        rows[p] = ra
        print(f"{p:12s} {ksb:10.3f} {ksa:10.3f} {bb:+8.3f} {ba:+8.3f}")

    np.savez(args.out_map, params=np.array(list(maps)),
             **{f'g_{p}': g for p, g in maps.items()})
    print(f"saved {args.out_map}")
    # recalibrated held-out test ranks (for the bias/coverage summary)
    np.savez(product_out('pp_ranks_recal.npz'), N=int(test['N']),
             n_ucb=int(test['n_ucb']) if 'n_ucb' in test.files else 20, **rows)
    print(f"saved {product_out('pp_ranks_recal.npz')}")

    # --- recalibrated test-set figures ---
    from evaluation.plotstyle import paper_style, HERON_CORAL, CORNFLOWER
    import matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as plt
    paper_style()
    Ntest = int(test['N'])
    order = [p for p in ['Mc', 'q', 'c1', 'c2', 'logD', 'cos_inc', 'sin_beta',
                         'lambda', 'psi', 'phi0', 't_pre_hours'] if p in rows]
    LBL = {'Mc': r'$\mathcal{M}_c$', 'q': r'$q$', 'c1': r'$\chi_1$', 'c2': r'$\chi_2$',
           'logD': r'$\log D_L$', 'cos_inc': r'$\cos\iota$', 'sin_beta': r'$\sin\beta$',
           'lambda': r'$\lambda$', 'psi': r'$\psi$', 'phi0': r'$\phi_0$',
           't_pre_hours': r'$t_c$'}
    PAL = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
           '#e377c2', '#17becf', '#bcbd22', '#7f7f7f', '#393b79']
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    xp = np.linspace(0, 1, 200); sci = np.sqrt(xp * (1 - xp) / Ntest)
    for k, al in zip((1, 2, 3), (0.18, 0.12, 0.07)):
        ax.fill_between(xp, np.clip(xp - k * sci, 0, 1), np.clip(xp + k * sci, 0, 1),
                        color='0.5', alpha=al, lw=0)
    yv = np.arange(1, Ntest + 1) / Ntest
    for p, c in zip(order, PAL):
        pv = kstest(rows[p], 'uniform').pvalue
        ax.plot(np.sort(rows[p]), yv, color=c, lw=1.6, label=f"{LBL[p]}  ($p={pv:.2f}$)")
    ax.plot([0, 1], [0, 1], 'k', ls=(0, (4, 3)), lw=1.1, alpha=0.7)
    ax.set_xlabel('Theoretical CDF'); ax.set_ylabel('Empirical CDF')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_title('HERON calibration (recalibrated, held-out test)')
    ax.legend(fontsize=8.5, ncol=2, loc='upper left', frameon=False)
    ax.text(0.97, 0.03, f'$N={Ntest}$ held-out injections', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8, color='0.35')
    fig.savefig(figure_out('fig_pp.pdf'), bbox_inches='tight')
    pv = {p: kstest(rows[p], 'uniform').pvalue for p in order}
    print(f"KS p-values after recalibration: min {min(pv.values()):.2f}  max {max(pv.values()):.2f}")
    print(f"saved {figure_out('fig_pp.pdf')}  (Fig. 3a)")


if __name__ == '__main__':
    main()
