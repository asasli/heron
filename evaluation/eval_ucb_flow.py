"""
UCB flow validation: P-P calibration + per-param accuracy + THROUGHPUT.

Throughput is the Nature headline: time to produce posteriors for the full
resolved galactic-binary catalog (~1e4 sources) vs MCMC (~hours each).

Run: python evaluation/eval_ucb_flow.py
"""
import sys, json, time, argparse
import numpy as np
import torch
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from heron.artifacts import (MBHB_FLOW, MBHB_NORM, MBHB_RECAL, UCB_FLOW, UCB_RESPONSE,
                             MBHB_DATA, UCB_DATA, load_kernel, product_out,
                             product_in, figure_out)
from pe_ucb.train_flow import UCBFlow, augment_, norm_targets, RANGES, PARAMS, N_PAR
from evaluation.plotstyle import paper_style


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=UCB_FLOW)
    ap.add_argument('--data', default=UCB_DATA)
    ap.add_argument('--n-pp', type=int, default=4000)
    ap.add_argument('--no-neigh', action='store_true')
    ap.add_argument('--no-snr-cut', action='store_true')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    torch.manual_seed(0)              # deterministic flow draws
    device = torch.device(args.device)

    ck = torch.load(args.model, map_location=device, weights_only=False)
    cfg = ck['config']
    m = UCBFlow(W=cfg['W']).to(device)
    m.load_state_dict(ck['model_state_dict']); m.eval()
    sigma0 = cfg['sigma0']
    root = Path(args.data)
    feats = torch.from_numpy(np.load(root / 'test' / 'feats.npy'))
    md = json.loads((root / 'test' / 'metadata.json').read_text())
    tg = torch.from_numpy(norm_targets(md))
    cr = torch.tensor([x['conf_ratio'] for x in md], dtype=torch.float32)
    snr = np.array([x['snr'] for x in md])
    print(f"model ep{ck['epoch']} val_nll {ck['val_nll']:.3f} | test {len(feats)}")

    # realistic test conditions: noise + confusion + neighbors (fixed seed)
    g = torch.Generator(device=device).manual_seed(11)
    N = min(args.n_pp, len(feats))
    keep = (np.arange(len(snr)) if args.no_snr_cut else np.where(snr > 8)[0])[:N]
    X = feats[keep].to(device).clone()
    X = augment_(X, cr[keep].to(device), sigma0, rng=g, p_neigh=0.0 if args.no_neigh else 0.5) / sigma0
    Y = tg[keep].to(device)

    # P-P
    pct = {p: [] for p in PARAMS}
    with torch.no_grad():
        for a0 in range(0, len(X), 256):
            s = m.sample(X[a0:a0+256], 1000)          # (b, 1000, 7)
            t = Y[a0:a0+256]
            for j, p in enumerate(PARAMS):
                pct[p].extend(((s[:, :, j] <= t[:, j:j+1]).float().mean(1)).cpu().numpy())
    from scipy.stats import kstest
    print("KS p:", "  ".join(f"{p}={kstest(np.array(pct[p]),'uniform').pvalue:.2f}"
                             for p in PARAMS))

    # accuracy (posterior median error), bright subset SNR>50
    bright = np.where(snr[keep] > 50)[0][:1000]
    with torch.no_grad():
        s = m.sample(X[bright], 2000)
    med = s.median(dim=1).values.cpu().numpy()
    tt = Y[bright].cpu().numpy()
    lo, hi = RANGES['df_bins']
    df_err = np.abs((med[:, 1] - tt[:, 1]) * (hi - lo))
    print(f"frequency precision (SNR>50): median |df| = {np.median(df_err):.3f} bins "
          f"= {np.median(df_err)*3.803e-7*1e9:.2f} nHz")

    # throughput
    with torch.no_grad():
        _ = m.sample(X[:64], 100)                      # warmup
        torch.cuda.synchronize(); t0 = time.time()
        BT, NS = 512, 1000
        for a0 in range(0, 4096, BT):
            _ = m.sample(X[a0:a0+BT], NS)
        torch.cuda.synchronize()
        dt = time.time() - t0
    per = dt / 4096
    print(f"throughput: {per*1e3:.2f} ms/source ({NS} samples) -> "
          f"10^4 sources in {per*1e4:.0f} s = {per*1e4/60:.1f} min")

    # P-P figure (paper)
    paper_style()
    LBL = {'log10A': r'$\log_{10}\!A$', 'df_bins': r'$\delta f$', 'phi0': r'$\phi_0$',
           'cos_iota': r'$\cos\iota$', 'psi': r'$\psi$', 'lambda': r'$\lambda$',
           'sin_beta': r'$\sin\beta$'}
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    xp = np.linspace(0, 1, 200); sci = np.sqrt(xp*(1-xp)/len(keep))
    for k, a in zip((1, 2, 3), (0.18, 0.12, 0.07)):
        ax.fill_between(xp, np.clip(xp-k*sci, 0, 1), np.clip(xp+k*sci, 0, 1),
                        color='0.5', alpha=a, lw=0)
    y = np.arange(1, len(keep)+1) / len(keep)
    for p, c in zip(PARAMS, plt.cm.tab10(np.linspace(0, 1, N_PAR))):
        pv = kstest(np.array(pct[p]), 'uniform').pvalue
        ax.plot(np.sort(pct[p]), y, lw=1.6, color=c, label=f"{LBL[p]} ($p={pv:.2f}$)")
    ax.plot([0, 1], [0, 1], 'k', ls=(0, (4, 3)), lw=1, alpha=0.7)
    ax.set_xlabel('Theoretical CDF'); ax.set_ylabel('Empirical CDF')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_title('Galactic-binary PE calibration (noise + confusion + neighbours)')
    ax.legend(fontsize=9, frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(figure_out('fig_ucb_pp.png'), dpi=300, bbox_inches='tight')
    fig.savefig(figure_out('fig_ucb_pp.pdf'), bbox_inches='tight')
    print(f"saved {figure_out('fig_ucb_pp.pdf')}")


if __name__ == '__main__':
    main()
