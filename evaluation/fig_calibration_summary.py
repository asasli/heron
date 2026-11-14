"""MBHB calibration summary: separates BIAS from WIDTH — the two things a
single P-P curve conflates.

From the per-event percentile ranks (pp_ranks_recal.npz, written by
evaluation/fit_recalibration.py; the published file is in paper_products/):
  (a) bias      = mean rank - 0.5 per parameter, with the binomial null band;
                  consistent with 0 => no systematic offset (unbiased).
  (b) coverage  = fraction of truths inside the nominal 50% and 90% credible
                  intervals; matches nominal => correctly-sized posteriors.

Out: results/figures/fig_calibration_summary.pdf  (Fig. 3b,c)
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.plotstyle import paper_style, HERON_CORAL, CORNFLOWER
from heron.artifacts import product_in, figure_out

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--ranks', default=None, help='rank file (default: pp_ranks_recal.npz product)')
_args, _ = _ap.parse_known_args()
RANKS = _args.ranks or product_in('pp_ranks_recal.npz')
OUT = figure_out('fig_calibration_summary.pdf')
ORDER = ['Mc', 'q', 'chi_eff', 'c1', 'c2', 'logD', 'cos_inc', 'sin_beta',
         'lambda', 'psi', 'phi0', 't_pre_hours']
LBL = {'Mc': r'$\mathcal{M}_c$', 'q': r'$q$', 'chi_eff': r'$\chi_{\rm eff}$',
       'c1': r'$\chi_1$', 'c2': r'$\chi_2$', 'logD': r'$\log D_L$',
       'cos_inc': r'$\cos\iota$', 'sin_beta': r'$\sin\beta$',
       'lambda': r'$\lambda$', 'psi': r'$\psi$', 'phi0': r'$\phi_0$',
       't_pre_hours': r'$t_c$'}


def main():
    paper_style()
    z = np.load(RANKS)
    N = int(z['N'])
    names = [n for n in ORDER if n in z.files]
    y = np.arange(len(names))[::-1]

    bias = np.array([np.mean(z[n]) - 0.5 for n in names])
    bias_err = np.sqrt(1 / (12 * N))                      # uniform-null SE of the mean
    cov50 = np.array([np.mean((z[n] > 0.25) & (z[n] < 0.75)) for n in names])
    cov90 = np.array([np.mean((z[n] > 0.05) & (z[n] < 0.95)) for n in names])
    err50 = np.sqrt(0.5 * 0.5 / N); err90 = np.sqrt(0.9 * 0.1 / N)

    fig, ax = plt.subplots(1, 2, figsize=(8.4, 4.4), gridspec_kw=dict(wspace=0.35))

    # --- (a) bias ---
    a = ax[0]
    a.axvspan(-2 * bias_err, 2 * bias_err, color='0.85', lw=0, label=r'$\pm2\sigma$ null')
    a.axvline(0, color='k', lw=0.8)
    a.errorbar(bias, y, xerr=2 * bias_err, fmt='o', color=HERON_CORAL, ms=6,
               capsize=2, lw=1)
    a.set_yticks(y); a.set_yticklabels([LBL[n] for n in names])
    a.set_xlabel(r'bias:  $\langle\,$rank$\,\rangle - 0.5$')
    a.set_title('(a)  no systematic offset', fontsize=10)
    a.legend(fontsize=8, frameon=False, loc='lower right')
    xm = max(0.04, 1.2 * np.abs(bias).max()); a.set_xlim(-xm, xm)

    # --- (b) coverage ---
    b = ax[1]
    b.errorbar(cov90 * 100, y + 0.12, xerr=2 * err90 * 100, fmt='s',
               color=CORNFLOWER, ms=5, capsize=2, lw=1, label=r'$90\%$ CI')
    b.errorbar(cov50 * 100, y - 0.12, xerr=2 * err50 * 100, fmt='o',
               color=HERON_CORAL, ms=5, capsize=2, lw=1, label=r'$50\%$ CI')
    b.axvline(90, color=CORNFLOWER, ls='--', lw=1, alpha=0.7)
    b.axvline(50, color=HERON_CORAL, ls='--', lw=1, alpha=0.7)
    b.set_yticks(y); b.set_yticklabels([])
    b.set_xlabel('empirical coverage [\\%]')
    b.set_title('(b)  correctly-sized intervals', fontsize=10)
    b.set_xlim(35, 100); b.legend(fontsize=8, frameon=False, loc='lower left')

    fig.suptitle(rf'HERON calibration, $N={N}$ injections with galactic foreground',
                 fontsize=11, y=1.0)
    fig.savefig(OUT, bbox_inches='tight', dpi=200)
    print("bias (rank-0.5):", {n: round(float(np.mean(z[n]) - 0.5), 3) for n in names})
    print("cov50:", {n: round(float(c), 2) for n, c in zip(names, cov50)})
    print("cov90:", {n: round(float(c), 2) for n, c in zip(names, cov90)})
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
