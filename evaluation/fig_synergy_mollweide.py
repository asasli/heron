"""Fig. 5a — full-sky Mollweide of the HERON-subtracted HyperWave posterior,
in the dark-ellipse pipeline style (magma density on a dark sky, waveform-
mapped injection star). Matches the stage-4 thumbnail of Fig. 1a.

Data: pointing/synergy_subtracted_alert.npz (stage-4 chain; pointing/synergy_pe.py).
Out: results/figures/fig_synergy_mollweide.pdf
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
from evaluation.plotstyle import paper_style

OUT = figure_out('fig_synergy_mollweide.pdf')
I_SB, I_LAM = 7, 8
TRUTH_LAM, TRUTH_SB = 5.0639, -0.2927        # waveform-mapped injection


def wrap(lam):
    return ((lam + np.pi) % (2 * np.pi)) - np.pi


def main():
    paper_style()
    z = np.load(product_in('pointing/synergy_subtracted_alert.npz'), allow_pickle=True)
    s = z['samples']
    lon = wrap(s[:, I_LAM].astype(float))
    lat = np.arcsin(np.clip(s[:, I_SB].astype(float), -1, 1))

    nlon, nlat = 360, 180
    H, xe, ye = np.histogram2d(lon, np.sin(lat), bins=[nlon, nlat],
                               range=[[-np.pi, np.pi], [-1, 1]])
    H = gaussian_filter(H.T, 1.6, mode=('nearest', 'wrap'))
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = np.arcsin(np.clip(0.5 * (ye[:-1] + ye[1:]), -1, 1))
    X, Y = np.meshgrid(xc, yc)

    flat = np.sort(H.ravel())[::-1]
    csum = np.cumsum(flat) / flat.sum()
    lev90, lev50 = (flat[np.searchsorted(csum, q)] for q in (0.90, 0.50))

    fig = plt.figure(figsize=(6.4, 3.6))
    ax = fig.add_subplot(111, projection='mollweide')
    ax.set_facecolor('#0b0518')                     # dark sky inside the ellipse
    ax.pcolormesh(X, Y, H, cmap='magma', shading='gouraud', rasterized=True,
                  vmin=0, vmax=H.max())
    tl, tb = wrap(np.array([TRUTH_LAM]))[0], np.arcsin(TRUTH_SB)
    al = wrap(np.array([TRUTH_LAM + np.pi]))[0]
    ax.plot(tl, tb, marker='*', ms=15, mfc='white', mec='k', mew=1.0,
            ls='none', zorder=6)
    ax.annotate('dominant', xy=(tl, tb), xytext=(tl - 0.02, tb - 0.62),
                fontsize=8, color='white', ha='center',
                arrowprops=dict(arrowstyle='-', color='0.8', lw=0.5))
    ax.annotate('antipodal', xy=(al, -tb), xytext=(al + 0.02, -tb + 0.58),
                fontsize=8, color='white', ha='center',
                arrowprops=dict(arrowstyle='-', color='0.8', lw=0.5))

    ax.grid(True, lw=0.25, color='white', alpha=0.22)
    ax.tick_params(labelsize=6, colors='0.45')
    ax.set_yticklabels([f'{int(np.degrees(t))}°' for t in ax.get_yticks()],
                       fontsize=6, color='0.45')
    ax.set_xticklabels([])
    ax.set_title(r'HERON-subtracted sky posterior '
                 r'($A_{90}=143\,$deg$^2$, both sky modes)', fontsize=9, pad=6)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches='tight', dpi=300)
    print(f'saved {OUT}')


if __name__ == '__main__':
    main()
