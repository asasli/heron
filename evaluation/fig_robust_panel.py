"""Fig. 3b — foreground robustness: per-event 90% sky area vs n_UCB.

Data: robust_sweep.npz (evaluation/eval_robust.py: released model, 250 events,
      4 GNPE rounds; the published file is in paper_products/).
Out:  results/figures/fig_robust_panel.pdf
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from heron.artifacts import product_in, product_out, figure_out
from evaluation.plotstyle import paper_style, HERON_CORAL

ROOT = Path(__file__).resolve().parent.parent
z = np.load(product_in('robust_sweep.npz'))
NU = [0, 10, 20, 40]

paper_style()
fig, ax = plt.subplots(figsize=(4.6, 4.2))
rng = np.random.default_rng(3)
for i, n in enumerate(NU):
    a = z[f'area_{n}']
    x = i + rng.uniform(-0.16, 0.16, len(a))
    ax.scatter(x, a, s=5, color='0.62', alpha=0.5, lw=0, rasterized=True)
    med = np.median(a)
    ax.hlines(med, i - 0.27, i + 0.27, color=HERON_CORAL, lw=2.6, zorder=5)
    ax.text(i, med * 1.22, f'{med:.0f}', ha='center', fontsize=9,
            color=HERON_CORAL)
ax.plot([], [], color=HERON_CORAL, lw=2.6, label='median')
ax.scatter([], [], s=10, color='0.55', label='events (250)')
ax.set_yscale('log')
ax.set_xticks(range(len(NU)))
ax.set_xticklabels([str(n) for n in NU])
ax.set_xlabel(r'resolved foreground sources $n_{\rm UCB}$')
ax.set_ylabel(r'$90\%$ sky area [deg$^2$]')
ax.legend(loc='lower right', frameon=False, fontsize=9)
fig.tight_layout()
OUT = figure_out('fig_robust_panel.pdf')
fig.savefig(OUT, bbox_inches='tight')
print(f'saved {OUT}')
