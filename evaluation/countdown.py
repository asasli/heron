"""
LOCALIZATION COUNTDOWN — the multi-messenger figure for Nature Astronomy.

(a) Population: median 90% sky area vs time-before-merger (t_pre bins of the
    wide-spin v7 test set), with LSST / Athena-WFI fields of view overlaid.
(b) Single source (Case 1) tracked as the merger approaches: the same event
    evaluated at t_pre = 600 -> 1 h (data regenerated per epoch, bbhx).

Run: python evaluation/countdown.py
"""
import sys, argparse, json
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
import cupy as cp
from pe_smbhb.data_mbhb import HetDataGenerator
from pe_smbhb.train_flow import (build_model, HetDataset, sky_pose_meta, ref_meta_vec,
                              sky_area_90, SKY_SIGMA_MIN, SKY_SIGMA_MAX)
from pe_smbhb.inference import draw_ref, het_features, gnpe_marg
from evaluation.plotstyle import paper_style, HERON_CORAL, CORNFLOWER
from pe_smbhb.inference import circ_median
from heron.sources import SANGRIA as CASES

LSST_FOV = 9.6      # deg^2
ATHENA_FOV = 0.4    # deg^2 (WFI)
TPRE_BINS = [(0.5, 2), (2, 6), (6, 24), (24, 72), (72, 200), (200, 600)]


def gnpe_area(m, fd, base, device, rounds=3, inflate=1.6, n_post=3000, k_pose=6, seed=0):
    sb, lam, sig = 0.5, 0.5, SKY_SIGMA_MAX
    g = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        for r in range(rounds - 1):
            pose = sky_pose_meta(torch.tensor([[sb, lam]], device=device, dtype=torch.float32),
                                 torch.tensor([sig], device=device, dtype=torch.float32), gen=g)
            s_ = m.sample(fd, n_post, meta=torch.cat([base, pose], -1)).cpu().numpy().squeeze()
            s_ = np.clip(s_, 1e-4, 1-1e-4)
            sb = float(np.median(s_[:, 6])); lam = circ_median(s_[:, 7])
            sig = float(np.clip(max(s_[:, 6].std(), s_[:, 7].std()) * inflate,
                                SKY_SIGMA_MIN, SKY_SIGMA_MAX))
        outs = []
        for _ in range(k_pose):
            pose = sky_pose_meta(torch.tensor([[sb, lam]], device=device, dtype=torch.float32),
                                 torch.tensor([sig], device=device, dtype=torch.float32), gen=g)
            o = m.sample(fd, n_post // k_pose + 1, meta=torch.cat([base, pose], -1)).cpu().numpy().squeeze()
            outs.append(np.clip(o, 1e-4, 1-1e-4))
    samp = np.concatenate(outs, 0)
    return sky_area_90(samp[:4000, 7] * 2*np.pi, samp[:4000, 6] * 2 - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=MBHB_FLOW)
    ap.add_argument('--norm', default=MBHB_NORM)
    ap.add_argument('--data', default=MBHB_DATA)
    ap.add_argument('--max-per-bin', type=int, default=120)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    torch.manual_seed(0)              # deterministic flow draws
    device = torch.device(args.device)

    ck = torch.load(args.model, map_location=device, weights_only=False)
    cfg = ck['config']
    m = build_model(cfg).to(device); m.load_state_dict(ck['model_state_dict']); m.eval()
    nm = torch.load(args.norm, map_location=device, weights_only=False)
    fmean, fstd = nm['fd_mean'].to(device), nm['fd_std'].to(device)

    # panel (a): population, stored wide-spin test set
    ds = HetDataset(args.data, 'test',
                    residual=cfg.get('residual_params') or cfg.get('residual_targets', True))
    tp = np.array([mm['smbhb']['t_pre_hours'] for mm in ds.metadata])
    snr = np.array([mm['smbhb']['snr'] for mm in ds.metadata])
    rows = {b: [] for b in TPRE_BINS}
    rng = np.random.default_rng(0)
    for b in TPRE_BINS:
        idxs = np.where((tp >= b[0]) & (tp < b[1]) & (snr > 30))[0]
        rng.shuffle(idxs)
        for i in idxs[:args.max_per_bin]:
            e = ds[int(i)]
            fd = (e['fd'].unsqueeze(0).to(device) - fmean) / fstd
            base = e['meta'].unsqueeze(0).to(device)
            if cfg.get('ucb_cond', False):
                base = torch.cat([base, torch.zeros(1, 1, device=device)], -1)
            rows[b].append(gnpe_area(m, fd, base, device, seed=int(i)))
        print(f"bin {b}: n={len(rows[b])}, median {np.median(rows[b]):.0f} deg^2", flush=True)

    # panel (b): Case-1 source tracked toward merger
    gen = HetDataGenerator(output_dir=None)
    np.random.seed(123)
    for _ in range(2):
        Pw, _ = gen.draw_priors(64); gen.gen_wf(Pw)
    kbase = load_kernel(args.data)
    S = CASES[1]['src']
    m1, m2 = S['m1'], S['m2']; eta = m1*m2/(m1+m2)**2
    Mc = (m1+m2)*eta**0.6; q = m1/m2
    TPRE_TRACK = [600, 300, 100, 24, 6, 1]
    track = []
    for tpre in TPRE_TRACK:
        t_ref = (gen.t_obs - tpre*3600)/3600
        pt = np.array([[Mc, q, S['chi1'], S['chi2'], np.log10(S['dist_Mpc']), t_ref,
                        np.cos(S['iota']), np.sin(S['beta']), S['lam'], S['psi'], S['phi0']]])
        A, E = gen.gen_wf(pt)
        rg = cp.random.RandomState(1)
        dA = A[0]/gen.sqrt_psd_g + (rg.standard_normal(gen.n_bins)+1j*rg.standard_normal(gen.n_bins))*gen.sigma0
        dE = E[0]/gen.sqrt_psd_g + (rg.standard_normal(gen.n_bins)+1j*rg.standard_normal(gen.n_bins))*gen.sigma0
        snr1 = float(cp.sqrt(4*gen.df*cp.sum((cp.abs(A[0])**2+cp.abs(E[0])**2)/gen.psd_g)))
        ref = draw_ref(pt[0], tpre, kbase, np.random.RandomState(7), s=1.0, wide_spin=True)
        feats = het_features(gen, dA, dE, ref)
        fd = (torch.as_tensor(cp.asnumpy(feats), dtype=torch.float32, device=device).unsqueeze(0)
              - fmean) / fstd
        base = torch.from_numpy(ref_meta_vec(ref)).unsqueeze(0).to(device).float()
        if cfg.get('ucb_cond', False):
            base = torch.cat([base, torch.zeros(1, 1, device=device)], -1)
        a = gnpe_area(m, fd, base, device, n_post=6000)
        track.append((tpre, a, snr1))
        print(f"track t_pre={tpre}h: SNR {snr1:.0f}, area {a:.0f} deg^2", flush=True)

    # figure
    paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    ctr = [np.sqrt(b[0]*b[1]) for b in TPRE_BINS]
    med = [np.median(rows[b]) for b in TPRE_BINS]
    q1 = [np.percentile(rows[b], 25) for b in TPRE_BINS]
    q3 = [np.percentile(rows[b], 75) for b in TPRE_BINS]
    ax.fill_between(ctr, q1, q3, color=HERON_CORAL, alpha=0.20, lw=0)
    ax.plot(ctr, med, 'o-', color=HERON_CORAL, lw=2, ms=5, label='HERON median (IQR)')
    for y, lbl in [(LSST_FOV, 'Rubin/LSST FoV'), (ATHENA_FOV, 'Athena WFI FoV')]:
        ax.axhline(y, color='0.35', ls=':', lw=1.1)
        ax.text(ctr[0], y*1.25, lbl, fontsize=8.5, color='0.35')
    ax.set(xscale='log', yscale='log', xlabel='time before merger [h]',
           ylabel=r'90\% credible sky area [deg$^2$]' if plt.rcParams['text.usetex']
                  else '90% credible sky area [deg$^2$]')
    ax.invert_xaxis()
    ax.set_title('(a) population')
    ax.legend(frameon=False, loc='lower left', fontsize=9)

    ax = axes[1]
    tt = [t for t, _, _ in track]; aa = [a for _, a, _ in track]
    ax.plot(tt, aa, 'o-', color=CORNFLOWER, lw=2, ms=6)
    for (t, a, s_) in track:
        ax.annotate(f'SNR {s_:.0f}', (t, a), textcoords='offset points',
                    xytext=(6, 6), fontsize=7.5, color='0.4')
    for y, lbl in [(LSST_FOV, 'Rubin/LSST'), (ATHENA_FOV, 'Athena WFI')]:
        ax.axhline(y, color='0.35', ls=':', lw=1.1)
    ax.set(xscale='log', yscale='log', xlabel='time before merger [h]')
    ax.invert_xaxis()
    ax.set_title('(b) single source approaching merger')
    fig.suptitle('HERON localization countdown (inference $<1$ s per epoch)'
                 if plt.rcParams['text.usetex'] else
                 'HERON localization countdown (inference <1 s per epoch)', y=1.02)
    fig.tight_layout()
    fig.savefig(figure_out('fig_countdown.png'), dpi=300, bbox_inches='tight')
    fig.savefig(figure_out('fig_countdown.pdf'), bbox_inches='tight')
    np.save(product_out('countdown_data.npy'),
            dict(bins=TPRE_BINS, rows={str(b): rows[b] for b in TPRE_BINS}, track=track),
            allow_pickle=True)
    print(f"saved {product_out('countdown_data.npy')} and {figure_out('fig_countdown.pdf')}")


if __name__ == '__main__':
    main()
