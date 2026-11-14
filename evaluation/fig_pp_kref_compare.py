"""Overlay P-P (rank-CDF) for K_ref=1 vs K_ref=16 from saved rank files, so the
intrinsic over-dispersion of reference marginalization is visible: a calibrated
parameter tracks the diagonal; an over-dispersed (too-wide/conservative) one
bows ABOVE the diagonal at low p and below at high p (S inside the band); the
rank-std is annotated (uniform = 0.289)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from heron.artifacts import product_in, figure_out
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

R1 = np.load(product_in('pp_ranks.npz'))       # K_ref=1
R16 = np.load(product_in('pp_ranks_k16.npz'))  # K_ref=16 (pp_marginalized.py --k-ref 16 --out-ranks .../pp_ranks_k16.npz)
N1, N16 = int(R1['N']), int(R16['N'])
PARAMS = ['Mc', 'q', 'chi_eff', 'c1', 'c2', 'logD', 'sin_beta', 'lambda']
LAB = {'Mc': r'$\mathcal{M}_c$', 'q': r'$q$', 'chi_eff': r'$\chi_{\rm eff}$',
       'c1': r'$\chi_1$', 'c2': r'$\chi_2$', 'logD': r'$\log D_L$',
       'sin_beta': r'$\sin\beta$', 'lambda': r'$\lambda$'}


def ppcurve(r):
    r = np.sort(np.asarray(r, float))
    return r, (np.arange(len(r)) + 0.5) / len(r)


fig, axes = plt.subplots(2, 4, figsize=(15, 7.6))
for ax, p in zip(axes.ravel(), PARAMS):
    r1 = R1[p].astype(float); r1 = r1 / N1 if r1.max() > 1.5 else r1
    r16 = R16[p].astype(float); r16 = r16 / N16 if r16.max() > 1.5 else r16
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6)
    # 2-sigma band for uniform
    m = len(r1)
    x = np.linspace(0, 1, 100)
    band = 2 * np.sqrt(x * (1 - x) / m)
    ax.fill_between(x, x - band, x + band, color='0.85', alpha=0.7, lw=0)
    for r, c, lab in [(r1, '#1f77b4', f'K=1  (std {r1.std():.2f})'),
                      (r16, '#d62728', f'K=16 (std {r16.std():.2f})')]:
        xs, ys = ppcurve(r)
        ax.plot(xs, ys, color=c, lw=1.8, label=lab)
    ax.set_title(LAB[p], fontsize=13)
    ax.legend(fontsize=8, loc='upper left', frameon=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
fig.suptitle('P-P: K$_{\\rm ref}$=1 (calibrated, on diagonal) vs K$_{\\rm ref}$=16 '
             '(intrinsics over-dispersed, bow off-diagonal)  |  uniform rank-std = 0.289',
             fontsize=12)
fig.supxlabel('nominal quantile', fontsize=11)
fig.supylabel('empirical rank-CDF', fontsize=11)
fig.tight_layout(rect=[0.01, 0.02, 1, 0.96])
out = figure_out('fig_pp_kref_compare.pdf')
fig.savefig(out, bbox_inches='tight')
fig.savefig(out.replace('.pdf', '.png'), dpi=140, bbox_inches='tight')
print('saved', out)
print(f"{'param':9s} {'K=1 std':>8s} {'K=16 std':>9s}  (uniform 0.289)")
for p in PARAMS:
    r1 = R1[p].astype(float); r1 = r1 / N1 if r1.max() > 1.5 else r1
    r16 = R16[p].astype(float); r16 = r16 / N16 if r16.max() > 1.5 else r16
    print(f"{p:9s} {r1.std():8.3f} {r16.std():9.3f}")
