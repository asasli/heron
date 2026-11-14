"""Pointing-tier ensemble (N=4 noise realizations x 3 datasets x 2 likelihoods).

Recomputes the AGREED canonical sky-area estimator (0.5-deg lambda x (2/360)
sin(beta) grid, Gaussian smoothing sigma=1 cell, greedy HPD count, BOTH modes
— identical to fig_synergy_sky.py) on every ensemble chain, prints the table,
and draws the ensemble strip plot: A90 per dataset, hyperbolic vs noise-fit
Gaussian likelihood (log scale; the Gaussian foreground blow-up is the point).

Data: the 24 stage-4 ensemble chains
      {hcl,hsg}_s{0..3}_{clean,foreground,subtracted}.npz
      (hcl = hyperbolic likelihood, hsg = noise-fit Gaussian; samples (300k,11)
      in HyperWave sampling order Mc q chi1 chi2 log10D tc cosi sinb lam psi phi).
      They are distributed as a separate archive (docs/DATA.md); unpack it to
      paper_products/pointing_ensemble/ or pass --chains <dir>. Without the
      chains the figure is drawn from the stored per-chain summary
      (pointing_ensemble_canonical.npz: A90, A50, correct-mode probability,
      truth-in-90%, peak distance, wall time for each of the 24 chains).

Run: python evaluation/fig_pointing_ensemble.py --hyp-only     (Fig. 5c; no GPU)
"""
import argparse
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from heron.artifacts import PAPER_PRODUCTS, product_in, product_out, figure_out
BASE = PAPER_PRODUCTS / 'pointing_ensemble'
NLAM, NSB, SM = 720, 360, 1.0
I_SB, I_LAM = 7, 8
DATASETS = ['clean', 'foreground', 'subtracted']
LIKES = {'hcl': 'hyperbolic', 'hsg': 'noise-fit Gaussian'}


# mapped injection in the HyperWave wrapper frame (waveform-match mapping,
# heron-hyperwave-handoff): lambda* = 290.14 deg, sin(beta)* = -0.2927
TRUTH_LAM, TRUTH_SB = 290.14, -0.2927


def areas(samples, burn=0.5):
    s = samples[int(len(samples)*burn):]
    lam = np.degrees(s[:, I_LAM].astype(float)) % 360.0
    sb = s[:, I_SB].astype(float)
    H, xe, ye = np.histogram2d(lam, sb, bins=[NLAM, NSB],
                               range=[[0., 360.], [-1., 1.]])
    Hs = gaussian_filter(H.T, SM, mode=('nearest', 'wrap'))
    flat = np.sort(Hs.ravel())[::-1]
    csum = np.cumsum(flat) / flat.sum()
    lev = [flat[np.searchsorted(csum, q)] for q in (0.90, 0.50)]
    cell = np.radians(xe[1]-xe[0]) * (ye[1]-ye[0]) * (180/np.pi)**2
    a90 = float(np.sum(Hs >= lev[0]) * cell)
    a50 = float(np.sum(Hs >= lev[1]) * cell)
    dom = float(((lam > 255) & (lam < 345) & (sb < 0.1)).mean())
    # truth coverage: is the mapped injection inside the 90% HPD region?
    it = np.searchsorted(xe, TRUTH_LAM) - 1
    jt = np.searchsorted(ye, TRUTH_SB) - 1
    in90 = bool(Hs[jt, it] >= lev[0])
    # angular distance of the density peak from the injection
    jj, ii = np.unravel_index(np.argmax(Hs), Hs.shape)
    plam = np.radians(0.5*(xe[ii]+xe[ii+1]))
    psb = 0.5*(ye[jj]+ye[jj+1])
    tl, tb = np.radians(TRUTH_LAM), np.arcsin(TRUTH_SB)
    pb = np.arcsin(np.clip(psb, -1, 1))
    dpk = np.degrees(np.arccos(np.clip(
        np.sin(tb)*np.sin(pb) + np.cos(tb)*np.cos(pb)*np.cos(tl-plam), -1, 1)))
    # antipodal image of the peak may be the matching mode
    dpk_ant = np.degrees(np.arccos(np.clip(
        np.sin(tb)*np.sin(-pb) + np.cos(tb)*np.cos(pb)*np.cos(tl-plam-np.pi), -1, 1)))
    return a90, a50, dom, in90, min(dpk, dpk_ant), dpk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--burn', type=float, default=0.0,
                    help='chains ship post-burn-in; 0.0 = use full stored chain')
    ap.add_argument('--hyp-only', action='store_true',
                    help='plot only the hyperbolic likelihood (paper Fig 5c); '
                         'the Gaussian areas alone mislead (stuck-tight modes '
                         'look small), so that comparison stays in the text')
    ap.add_argument('--sensitivity', action='store_true',
                    help='print burn-in sensitivity for one chain and exit')
    ap.add_argument('--chains', default=None, help='directory with the 24 ensemble chains')
    args = ap.parse_args()
    global BASE
    if args.chains:
        BASE = Path(args.chains)
    have_chains = (BASE / 'hcl_s0_clean.npz').exists()

    if args.sensitivity:
        z = np.load(BASE / 'hcl_s0_clean.npz')
        print(f"stored a90 curve final = {z['a90'][-1]:.0f}, a50 = {z['a50'][-1]:.0f}")
        for b in [0.0, 0.25, 0.5, 0.75, 0.9]:
            r = areas(z['samples'], burn=b)
            print(f"burn {b:.2f}: A90 {r[0]:7.0f}  A50 {r[1]:6.0f}  f_dom {r[2]:.2f}"
                  f"  in90 {r[3]}  dpeak {r[4]:.1f} deg")
        return

    res = {}
    if have_chains:
        print(f"recomputing the canonical estimator from the chains in {BASE}")
        for lk in LIKES:
            for ds in DATASETS:
                vals = []
                for s in range(4):
                    z = np.load(BASE / f'{lk}_s{s}_{ds}.npz')
                    a90, a50, dom, in90, dmin, dpk = areas(z['samples'], burn=args.burn)
                    vals.append((a90, a50, dom, float(in90), dmin, float(z['wall'])/60))
                res[(lk, ds)] = np.array(vals)
    else:
        z = np.load(product_in('pointing_ensemble_canonical.npz'))
        print(f"chains not found in {BASE}: using the stored per-chain summary")
        res = {(lk, ds): z[f'{lk}_{ds}'] for lk in LIKES for ds in DATASETS}

    print(f"canonical estimator, burn={args.burn}")
    print(f"{'like':10s} {'dataset':11s} {'A90 (4 seeds)':>28s} {'mean+-std':>14s} "
          f"{'in90':>10s} {'p(correct mode)':>26s}")
    for lk, nice in LIKES.items():
        for ds in DATASETS:
            v = res[(lk, ds)]
            a90s = v[:, 0]
            print(f"{nice:10s} {ds:11s} {np.array2string(a90s.round(0), precision=0):>28s} "
                  f"{a90s.mean():6.0f}+-{a90s.std():<5.0f} "
                  f"{np.array2string(v[:, 3].astype(int)):>10s} "
                  f"{np.array2string(v[:, 2].round(2)):>26s}")

    # figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from evaluation.plotstyle import paper_style, CORNFLOWER, HERON_CORAL
    VIOLET = "#9467BD"
    paper_style()
    COLORS = {'clean': CORNFLOWER, 'foreground': VIOLET, 'subtracted': HERON_CORAL}
    SINGLE = {'clean': 191, 'foreground': 134, 'subtracted': 143}

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xpos = {ds: i for i, ds in enumerate(DATASETS)}
    plot_likes = ['hcl'] if args.hyp_only else list(LIKES)
    OFF = {'hcl': 0.0 if args.hyp_only else -0.16, 'hsg': +0.16}
    rng = np.random.default_rng(1)
    for lk in plot_likes:
        for ds in DATASETS:
            v = res[(lk, ds)][:, 0]
            x = xpos[ds] + OFF[lk] + rng.uniform(-0.045, 0.045, len(v))
            filled = lk == 'hcl'
            ax.scatter(x, v, s=42, marker='o' if filled else 'D',
                       facecolor=COLORS[ds] if filled else 'none',
                       edgecolor=COLORS[ds], linewidth=1.4, zorder=3,
                       alpha=0.95)
            m = v.mean()
            ax.hlines(m, xpos[ds]+OFF[lk]-0.11, xpos[ds]+OFF[lk]+0.11,
                      color=COLORS[ds], lw=2.2 if filled else 1.4,
                      linestyle='-' if filled else (0, (3, 2)), zorder=4)
    for ds in DATASETS:
        ax.hlines(SINGLE[ds], xpos[ds]-0.34, xpos[ds]+0.34, color='0.25',
                  lw=1.0, linestyle=':', zorder=2)
    ax.set_yscale('log')
    ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels(['clean', 'foreground\n(unsubtracted)', 'HERON-\nsubtracted'])
    ax.set_ylabel(r'$90\%$ sky area [deg$^2$]')
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker='o', color='0.35', linestyle='-', lw=2,
                      markerfacecolor='0.35',
                      label='hyperbolic likelihood (per realization)')]
    if not args.hyp_only:
        handles.append(Line2D([], [], marker='D', color='0.35',
                              linestyle=(0, (3, 2)), lw=1.2,
                              markerfacecolor='none', label='noise-fit Gaussian'))
    handles.append(Line2D([], [], color='0.25', linestyle=':', lw=1,
                          label='single-realization run (panel b)'))
    ax.legend(handles=handles, loc='upper right', fontsize=9, frameon=False)
    ax.set_title('Pointing tier over noise realizations ($N=4$, alert-seeded)',
                 fontsize=11)
    fig.tight_layout()
    out = figure_out('fig_pointing_ensemble.pdf')
    fig.savefig(out, bbox_inches='tight')
    print('saved', out)
    if have_chains:
        np.savez(product_out('pointing_ensemble_canonical.npz'),
                 **{f'{lk}_{ds}': res[(lk, ds)] for lk in LIKES for ds in DATASETS})
        print('saved', product_out('pointing_ensemble_canonical.npz'))


if __name__ == '__main__':
    main()
