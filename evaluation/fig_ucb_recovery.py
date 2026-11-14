"""UCB unbiasedness: injected-vs-recovered (1:1) + standardized-residual (z)
histograms for the well-measured parameters (frequency and amplitude).

A 1:1 scatter along the diagonal shows no systematic offset (no bias); a z
histogram centred at 0 with unit width shows bias-free, correctly-sized
posteriors. Complements the P-P plot by separating bias from calibration.

Data: the held-out UCB test split (data/ucb/test) and the released UCB flow.
Out: results/figures/fig_ucb_recovery.pdf  (Extended Data Fig. 5)
"""
import sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from heron.artifacts import UCB_FLOW as MODEL, UCB_DATA as DATA, figure_out
from pe_ucb.train_flow import UCBFlow, augment_, norm_targets, RANGES, PARAMS
from evaluation.plotstyle import paper_style, HERON_CORAL

OUT = figure_out('fig_ucb_recovery.pdf')
DF_HZ = 3.803e-7          # bin width [Hz]


def main():
    paper_style()
    torch.manual_seed(0)
    device = torch.device('cuda:0')
    ck = torch.load(MODEL, map_location=device, weights_only=False)
    cfg = ck['config']
    m = UCBFlow(W=cfg['W']).to(device); m.load_state_dict(ck['model_state_dict']); m.eval()
    sigma0 = cfg['sigma0']
    root = Path(DATA)
    feats = torch.from_numpy(np.load(root / 'test' / 'feats.npy'))
    md = json.loads((root / 'test' / 'metadata.json').read_text())
    tg = torch.from_numpy(norm_targets(md))
    cr = torch.tensor([x['conf_ratio'] for x in md], dtype=torch.float32)
    snr = np.array([x['snr'] for x in md])

    g = torch.Generator(device=device).manual_seed(11)
    keep = np.where(snr > 8)[0][:3000]
    X = feats[keep].to(device).clone()
    X = augment_(X, cr[keep].to(device), sigma0, rng=g, p_neigh=0.5) / sigma0
    Y = tg[keep].cpu().numpy()
    snrk = snr[keep]

    meds, stds = [], []
    with torch.no_grad():
        for a0 in range(0, len(X), 256):
            s = m.sample(X[a0:a0 + 256], 1500).cpu().numpy()   # (b,1500,7)
            meds.append(np.median(s, axis=1)); stds.append(s.std(axis=1))
    med = np.concatenate(meds); std = np.concatenate(stds)

    def denorm(col, name):
        lo, hi = RANGES[name]
        return col * (hi - lo) + lo

    i_df, i_a = PARAMS.index('df_bins'), PARAMS.index('log10A')
    # frequency in nHz (df_bins -> Hz -> nHz), amplitude in log10A
    df_t = denorm(Y[:, i_df], 'df_bins') * DF_HZ * 1e9
    df_r = denorm(med[:, i_df], 'df_bins') * DF_HZ * 1e9
    df_s = std[:, i_df] * (RANGES['df_bins'][1] - RANGES['df_bins'][0]) * DF_HZ * 1e9
    a_t = denorm(Y[:, i_a], 'log10A'); a_r = denorm(med[:, i_a], 'log10A')
    a_s = std[:, i_a] * (RANGES['log10A'][1] - RANGES['log10A'][0])

    fig, ax = plt.subplots(2, 2, figsize=(8.2, 7.0))
    # --- 1:1 scatter: frequency ---
    sc = ax[0, 0].scatter(df_t, df_r, c=np.log10(snrk), s=6, cmap='viridis',
                          alpha=0.5, lw=0)
    lim = [min(df_t.min(), df_r.min()), max(df_t.max(), df_r.max())]
    ax[0, 0].plot(lim, lim, 'k--', lw=1, alpha=0.6)
    ax[0, 0].set_xlabel('injected $\\delta f$ [nHz]'); ax[0, 0].set_ylabel('recovered $\\delta f$ [nHz]')
    ax[0, 0].set_title('frequency recovery')
    cb = fig.colorbar(sc, ax=ax[0, 0]); cb.set_label(r'$\log_{10}$ SNR', fontsize=8)
    # --- 1:1 scatter: amplitude ---
    ax[0, 1].scatter(a_t, a_r, c=np.log10(snrk), s=6, cmap='viridis', alpha=0.5, lw=0)
    lim = [min(a_t.min(), a_r.min()), max(a_t.max(), a_r.max())]
    ax[0, 1].plot(lim, lim, 'k--', lw=1, alpha=0.6)
    ax[0, 1].set_xlabel(r'injected $\log_{10}A$'); ax[0, 1].set_ylabel(r'recovered $\log_{10}A$')
    ax[0, 1].set_title('amplitude recovery')
    # --- z histograms ---
    for axz, (t, r, sd, lab) in zip(
            (ax[1, 0], ax[1, 1]),
            [(df_t, df_r, df_s, r'$\delta f$'), (a_t, a_r, a_s, r'$\log_{10}A$')]):
        z = (r - t) / np.clip(sd, 1e-30, None)
        z = z[np.abs(z) < 6]
        axz.hist(z, bins=40, density=True, color=HERON_CORAL, alpha=0.7)
        xx = np.linspace(-5, 5, 200)
        axz.plot(xx, np.exp(-xx**2/2)/np.sqrt(2*np.pi), 'k-', lw=1.2, label=r'$\mathcal{N}(0,1)$')
        axz.set_xlabel(rf'$z = (\hat\theta-\theta_{{\rm true}})/\sigma$   [{lab}]')
        axz.set_title(rf'{lab}: mean $z={np.mean(z):.2f}$, std $={np.std(z):.2f}$')
        axz.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches='tight', dpi=200)
    print(f"freq: mean z={np.mean((df_r-df_t)/np.clip(df_s,1e-30,None)):.3f}")
    print(f"amp:  mean z={np.mean((a_r-a_t)/np.clip(a_s,1e-30,None)):.3f}")
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
