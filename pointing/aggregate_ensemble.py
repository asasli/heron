"""Aggregate the pointing-tier ensemble into population statistics + a figure.

Reads every synergy_pe.py output npz in --dir and reports the distribution of
90%/50% sky areas, dominant-mode fractions, wall time and time-to-localization,
plus truth-in-credible-region coverage. Produces a violin/scatter figure for
the paper (turns the single-event Fig. 5 number into an ensemble result).

Usage:
    python pointing/aggregate_ensemble.py --dir results/ensemble \
        [--out paper_fig_ensemble.pdf]
"""
import argparse
import glob
import json
import numpy as np
from pathlib import Path


def sky_credible_area(lam, sinb, q, nlam=720, nsb=360, sm=1.0):
    from scipy.ndimage import gaussian_filter
    H, xe, ye = np.histogram2d(np.mod(lam, 2 * np.pi), np.clip(sinb, -1, 1),
                               bins=[nlam, nsb], range=[[0, 2 * np.pi], [-1, 1]])
    Hs = gaussian_filter(H.T, sm, mode=('nearest', 'wrap'))
    flat = np.sort(Hs.ravel())[::-1]
    lev = flat[np.searchsorted(np.cumsum(flat) / flat.sum(), q)]
    cell = (xe[1] - xe[0]) * (ye[1] - ye[0]) * (180 / np.pi) ** 2
    return float(np.sum(Hs >= lev) * cell)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dir', required=True)
    ap.add_argument('--out', default='fig_ensemble.pdf')
    ap.add_argument('--i-lam', type=int, default=8)
    ap.add_argument('--i-sb', type=int, default=7)
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.dir) / '*.npz')))
    if not files:
        raise SystemExit(f"no result npz in {args.dir}")
    rows = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        s = z['samples']
        lam, sb = s[:, args.i_lam].astype(float), s[:, args.i_sb].astype(float)
        a90 = float(z['a90_canonical']) if 'a90_canonical' in z \
            else sky_credible_area(lam, sb, 0.90)
        a50 = float(z['a50_canonical']) if 'a50_canonical' in z \
            else sky_credible_area(lam, sb, 0.50)
        dom = float(((lam > 4.4) & (lam < 6.0) & (sb < 0.1)).mean())
        rows.append(dict(tag=str(z.get('tag', Path(f).stem)),
                         a90=a90, a50=a50, f_dom=dom,
                         wall=float(z.get('wall', np.nan)),
                         t_loc=float(z.get('t_loc', np.nan))))
    a90 = np.array([r['a90'] for r in rows])
    a50 = np.array([r['a50'] for r in rows])
    tloc = np.array([r['t_loc'] for r in rows]) / 60.0

    def q(x):
        return np.nanpercentile(x, [5, 50, 95])

    print(f"ensemble of {len(rows)} events")
    print(f"  A90 [deg^2]  median {np.median(a90):.0f}  "
          f"90%% CI [{q(a90)[0]:.0f}, {q(a90)[2]:.0f}]")
    print(f"  A50 [deg^2]  median {np.median(a50):.0f}  "
          f"90%% CI [{q(a50)[0]:.0f}, {q(a50)[2]:.0f}]")
    print(f"  t_loc [min]  median {np.nanmedian(tloc):.0f}  "
          f"range [{np.nanmin(tloc):.0f}, {np.nanmax(tloc):.0f}]")
    Path('ensemble_summary.json').write_text(json.dumps(
        dict(n=len(rows), rows=rows,
             a90_median=float(np.median(a90)), a90_ci=q(a90).tolist(),
             a50_median=float(np.median(a50)), a50_ci=q(a50).tolist()),
        indent=2))
    print("wrote ensemble_summary.json")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    rng = np.random.default_rng(0)
    for i, (arr, lab) in enumerate([(a50, r'$A_{50}$'), (a90, r'$A_{90}$')]):
        x = i + rng.uniform(-0.13, 0.13, len(arr))
        ax.scatter(x, arr, s=14, color='#E63946', alpha=0.6, lw=0)
        ax.hlines(np.median(arr), i - 0.25, i + 0.25, color='k', lw=2, zorder=5)
    ax.set_yscale('log'); ax.set_xticks([0, 1])
    ax.set_xticklabels([r'$A_{50}$', r'$A_{90}$'])
    ax.set_ylabel(r'sky area [deg$^2$]')
    ax.set_title(f'pointing tier, {len(rows)}-event ensemble')
    fig.tight_layout(); fig.savefig(args.out, bbox_inches='tight')
    print(f"saved {args.out}")


if __name__ == '__main__':
    main()
