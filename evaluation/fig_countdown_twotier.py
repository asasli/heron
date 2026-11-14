"""Countdown figure, two-tier version (replot from stored countdown_data.npy —
no GPU). Population alert envelope (median/IQR in purple-pink, best decile,
best event) PLUS the demonstrated stage-4 refinement (final HERON report:
HERON-seeded HyperWave, 143 deg^2 in ~20 min; N=4 ensemble scatter as error
bar) and the projected coherent-grid refinement (0.04-0.17 deg^2; Methods
feasibility study) that reaches the Athena single-pointing scale.

The operational bar for wide-field follow-up is NIGHTLY TILING CAPACITY, not a
single field of view: Rubin covers ~3x10^3 deg^2 per night (Ivezic et al.
2019), so the whole demonstrated tier is tileable in minutes-to-a-night.

Data: countdown_data.npy (evaluation/countdown.py; published file in paper_products/).
Run: python evaluation/fig_countdown_twotier.py   (no GPU)
"""
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from heron.artifacts import product_in, product_out, figure_out
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluation.plotstyle import paper_style, CORNFLOWER, HERON_CORAL

STAGE4_PURPLE = '#9467BD'               # stage-4 purple (matches Fig 1a schematic)
LSST_FOV, ATHENA_FOV = 9.6, 0.4
RUBIN_NIGHT = 3000.0                    # deg^2 tiled per night (Ivezic+2019)
STAGE4_T, STAGE4_A = 24.0, 143.0        # final HERON report (Fig 5)
ENS_MEAN, ENS_STD = 132.0, 82.0         # N=4 ensemble (subtracted, hyperbolic)
GRID_LO, GRID_HI = 0.04, 0.17           # coherent-grid refinement (projected)

d = np.load(product_in('countdown_data.npy'), allow_pickle=True).item()
bins = [tuple(b) for b in d['bins']]
rows = {tuple(eval(k)) if isinstance(k, str) else tuple(k): np.array(v)
        for k, v in d['rows'].items()}

paper_style()
fig, ax = plt.subplots(figsize=(7.6, 5.6))
ctr = [np.sqrt(b[0]*b[1]) for b in bins]
med = [np.median(rows[b]) for b in bins]
q1 = [np.percentile(rows[b], 25) for b in bins]
q3 = [np.percentile(rows[b], 75) for b in bins]
dec = [np.percentile(rows[b], 10) for b in bins]
best = [np.min(rows[b]) for b in bins]
ax.fill_between(ctr, q1, q3, color=HERON_CORAL, alpha=0.18, lw=0)
ax.plot(ctr, med, 'o-', color=HERON_CORAL, lw=2.2, ms=5.5,
        label='HERON alert: median (IQR)')
ax.plot(ctr, dec, 's--', color=CORNFLOWER, lw=1.8, ms=5, label='best decile')
ax.plot(ctr, best, 'v:', color='0.35', lw=1.6, ms=5, label='best event')

# final HERON report (demonstrated; Fig 5) + drop arrow from the envelope
med24 = float(np.interp(np.log(STAGE4_T), np.log(ctr[::-1]), np.array(med)[::-1]))
ax.annotate('', xy=(STAGE4_T, STAGE4_A*1.5), xytext=(STAGE4_T, med24/1.4),
            arrowprops=dict(arrowstyle='-|>', color='0.25', lw=1.4))
ax.text(STAGE4_T*0.55, STAGE4_A, 'seeded refinement,\n20 min',
        fontsize=9, color='0.25', ha='left', va='center')
ax.errorbar([STAGE4_T], [STAGE4_A],
            yerr=[[STAGE4_A-(ENS_MEAN-ENS_STD)], [(ENS_MEAN+ENS_STD)-STAGE4_A]],
            fmt='*', ms=17, color=STAGE4_PURPLE, mec='k', mew=0.8,
            ecolor='0.35', elinewidth=1.2, capsize=3, zorder=6,
            label='final HERON report')

# projected coherent-grid refinement (Methods feasibility): Athena scale
grid_mid = np.sqrt(GRID_LO*GRID_HI)
ax.annotate('', xy=(STAGE4_T, grid_mid*2.6), xytext=(STAGE4_T, STAGE4_A/1.8),
            arrowprops=dict(arrowstyle='-|>', color='0.45', lw=1.2, ls=(0, (4, 3))))
ax.errorbar([STAGE4_T], [grid_mid],
            yerr=[[grid_mid-GRID_LO], [GRID_HI-grid_mid]],
            fmt='*', ms=15, mfc='none', color='0.35', mec=STAGE4_PURPLE, mew=1.3,
            ecolor='0.45', elinewidth=1.1, capsize=3, zorder=6,
            label='coherent-grid refinement (projected; Methods)')

ax.axhspan(RUBIN_NIGHT, ax.get_ylim()[1] if False else 3.2e4, color='0.85',
           alpha=0.0)  # no-op keeps ylim stable
for y, lbl, style in [(RUBIN_NIGHT, r'Rubin nightly coverage ($\sim$3000 deg$^2$)', '--'),
                      (LSST_FOV, 'Rubin/LSST FoV (single pointing)', ':'),
                      (ATHENA_FOV, 'Athena WFI FoV', ':')]:
    ax.axhline(y, color='0.35', ls=style, lw=1.1)
    ytxt = y*0.58 if y == RUBIN_NIGHT else y*1.3   # nightly label below its line (clear space)
    ax.text(400, ytxt, lbl, fontsize=10.5, color='0.35', ha='left')
ylab = (r'90\% credible sky area [deg$^2$]' if plt.rcParams['text.usetex']
        else '90% credible sky area [deg$^2$]')
ax.set(xscale='log', yscale='log', xlabel='time before merger [h]', ylabel=ylab)
ax.set_ylim(1.3e-2, 4e4)
ax.invert_xaxis()
ax.legend(frameon=False, loc='lower left', fontsize=8.5)
ax.set_title('HERON localization countdown: continuous alert envelope + on-demand refinement',
             fontsize=10.5)
fig.tight_layout()
OUT = figure_out('fig_countdown.pdf')
fig.savefig(OUT, bbox_inches='tight')
fig.savefig(OUT.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
print(f'saved {OUT}')
