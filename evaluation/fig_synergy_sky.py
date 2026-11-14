"""Figure 5 (final) — HyperWave sky posteriors for the three HERON hand-off
data sets, with the waveform-geometry-mapped injection.

Area estimator = the agreed paper convention (HyperWave recommendation):
full-sky 2D histogram on a 0.5-deg lambda x (2/360) sin(beta) grid, Gaussian
smoothing sigma = 1 cell, greedy HPD cell count, BOTH sky islands included.
Reference values from the HyperWave side with this estimator:
A90 = 191/134/143 (clean/foreground/subtracted), A50 = 41/32/28, unseeded
772/109. Mapped truth (match 0.99997): lambda*=5.0639 rad, sin(beta)*=-0.2927.

Run: python evaluation/fig_synergy_sky.py
Data: pointing/synergy_{clean,foreground,subtracted}_alert.npz and
      synergy_subtracted_none.npz (stage-4 chains; pointing/synergy_pe.py).
Out: results/figures/fig_synergy_sky.pdf
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from heron.artifacts import product_in, product_out, figure_out
from evaluation.plotstyle import paper_style, CORNFLOWER, HERON_CORAL

OUT = figure_out('fig_synergy_sky.pdf')
I_SB, I_LAM = 7, 8
TRUTH = (np.degrees(5.0639), np.degrees(np.arcsin(-0.2927)))  # mapped injection
ZOOM = dict(lam=(255., 345.), beta=(-50., 5.))
VIOLET = '#9467BD'
COLORS = {'clean': CORNFLOWER, 'foreground': VIOLET, 'subtracted': HERON_CORAL}
STYLES = {'clean': 'solid', 'foreground': 'dashed'}
NLAM, NSB, SM = 720, 360, 1.0            # 0.5-deg grid, sigma = 1 cell


def sky_grid(tag, sm=SM):
    """Density + HPD levels. Areas ALWAYS at the agreed convention (sm=SM);
    pass a larger sm only for smoother DISPLAY contours."""
    z = np.load(product_in(f'pointing/synergy_{tag}.npz'), allow_pickle=True)
    s = z['samples']
    lam = np.degrees(s[:, I_LAM].astype(float))
    sb = s[:, I_SB].astype(float)
    H, xe, ye = np.histogram2d(lam, sb, bins=[NLAM, NSB],
                               range=[[0., 360.], [-1., 1.]])

    def levels(sigma):
        Hs = gaussian_filter(H.T, sigma, mode=('nearest', 'wrap'))
        flat = np.sort(Hs.ravel())[::-1]
        csum = np.cumsum(flat) / flat.sum()
        return Hs, [flat[np.searchsorted(csum, q)] for q in (0.90, 0.50)]

    Ha, lev_a = levels(SM)                    # agreed-convention areas
    cell = np.radians(xe[1] - xe[0]) * (ye[1] - ye[0]) * (180 / np.pi) ** 2
    a90 = float(np.sum(Ha >= lev_a[0]) * cell)
    a50 = float(np.sum(Ha >= lev_a[1]) * cell)
    Hs, lev = (Ha, lev_a) if sm == SM else levels(sm)
    xc, yc = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
    beta_c = np.degrees(np.arcsin(np.clip(yc, -1, 1)))
    dom = ((lam > 255) & (lam < 345) & (sb < 0.1)).mean()
    return xc, beta_c, Hs, lev, a90, a50, dom, z


def main():
    paper_style()
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    SM_DISP = 3.0                       # display smoothing (areas stay at SM)

    stats = {}
    # the hero first (filled), then the two comparisons as clean lines
    xc, bc, H, lev, a90, a50, dom, _ = sky_grid('subtracted_alert', sm=SM_DISP)
    ax.contourf(xc, bc, H, levels=[lev[0], lev[1], H.max()],
                colors=[HERON_CORAL], alpha=0.22)
    ax.contourf(xc, bc, H, levels=[lev[1], H.max()], colors=[HERON_CORAL], alpha=0.5)
    ax.contour(xc, bc, H, levels=lev, colors=HERON_CORAL, linewidths=[1.0, 1.6])
    stats['subtracted'] = (a90, a50, dom)
    for tag in ('foreground', 'clean'):
        xc, bc, H, lev, a90, a50, dom, _ = sky_grid(f'{tag}_alert', sm=SM_DISP)
        ax.contour(xc, bc, H, levels=lev, colors=COLORS[tag],
                   linewidths=[0.9, 1.4], alpha=0.9,
                   linestyles=STYLES[tag])
        stats[tag] = (a90, a50, dom)

    # unseeded 90% comparison, drawn faintly (full-sky view is panel a)
    xcu, bcu, Hu, levu, a90u, a50u, domu, _ = sky_grid('subtracted_none',
                                                       sm=SM_DISP)
    ax.contour(xcu, bcu, Hu, levels=[levu[0]], colors=['0.6'],
               linewidths=1.0, linestyles='dotted')

    ax.plot(*TRUTH, marker='*', ms=18, mfc='white', mec='k', mew=1.3,
            ls='none', zorder=6)

    for tag, label in [('subtracted', 'HERON-subtracted'),
                       ('foreground', 'foreground'), ('clean', 'clean')]:
        a90, a50, dom = stats[tag]
        ax.plot([], [], color=COLORS[tag], lw=2.4,
                ls=STYLES.get(tag, 'solid'),
                label=(rf"{label}:\ \ {a90:.0f}\,/\,{a50:.0f}$\,$deg$^2$,"
                       rf"\ \ $f_{{\rm dom}}={dom:.2f}$"))
        print(f"{tag}: A90={a90:.0f} A50={a50:.0f} f_dom={dom:.2f}")
    print(f"unseeded: A90={a90u:.0f} A50={a50u:.0f} f_dom={domu:.2f}")
    ax.plot([], [], color='0.6', ls='dotted', lw=1.0,
            label=rf"unseeded:\ \ {a90u:.0f}$\,$deg$^2$ (90\%)")
    ax.plot([], [], marker='*', ms=13, mfc='white', mec='k', mew=1.1,
            ls='none', label='injection (waveform-mapped)')

    ax.set_xlim(*ZOOM['lam']); ax.set_ylim(*ZOOM['beta'])
    ax.set_xlabel(r'ecliptic longitude $\lambda$ [deg]')
    ax.set_ylabel(r'ecliptic latitude $\beta$ [deg]')
    leg = ax.legend(loc='lower left', frameon=False, fontsize=9.5,
                    title=r'alert-seeded PE:\ \ $A_{90}\,/\,A_{50}$, both sky modes',
                    title_fontsize=9.5)
    leg._legend_box.align = 'left'

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches='tight')
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
