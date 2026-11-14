"""
Foreground-robustness sweep for the MBHB flow: iterative sky-GNPE localization with
an injected UCB foreground, as a function of n_ucb (resolved count) and the
faint-foreground level. If sky area + coverage stay flat across n_ucb, the
model is robust to the galactic foreground (the goal).

Run: python evaluation/eval_robust.py \
        --model weights/mbhb_flow.pt
"""
import sys, argparse
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from heron.artifacts import (MBHB_FLOW, MBHB_NORM, MBHB_RECAL, UCB_FLOW, UCB_RESPONSE,
                             MBHB_DATA, UCB_DATA, load_kernel, product_out,
                             product_in, figure_out)
from pe_smbhb.train_flow import (build_model, FlowHet, HetDataset, sky_area_90, sky_pose_meta,
                              inject_ucb_foreground, SKY_SIGMA_MIN, SKY_SIGMA_MAX,
                              UCB_N_MAX_COND)


def circ_median(l01):
    a = l01 * 2 * np.pi
    return (np.angle(np.mean(np.exp(1j * a))) % (2 * np.pi)) / (2 * np.pi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=MBHB_FLOW)
    ap.add_argument('--norm', default=MBHB_NORM)
    ap.add_argument('--data', default=MBHB_DATA)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--n-post', type=int, default=2000)
    ap.add_argument('--n-events', type=int, default=250)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    torch.manual_seed(0)              # deterministic flow draws
    device = torch.device(args.device)

    ck = torch.load(args.model, map_location=device, weights_only=False)
    cfg = ck['config']
    m = build_model(cfg).to(device)
    m.load_state_dict(ck['model_state_dict']); m.eval()
    print(f"model epoch {ck['epoch']} val_sky {ck.get('val_sky'):.3f}")
    nm = torch.load(args.norm, map_location=device, weights_only=False)
    fmean, fstd = nm['fd_mean'].to(device), nm['fd_std'].to(device)
    sig_bins = HetDataset(args.data, 'test', residual=True)  # also gives sigma_bins
    ds = sig_bins
    sigma_d = ds.sigma_bins.to(device)
    resp = np.load(UCB_RESPONSE)
    g_bin = torch.from_numpy(resp['g_bin'].astype(np.float32)).to(device)
    band = torch.from_numpy(np.where(resp['g_bin'] > 0)[0].astype(np.int64)).to(device)

    idxs = np.linspace(0, len(ds) - 1, args.n_events).astype(int)
    N_UCB_GRID = [0, 10, 20, 40]
    res = {nu: {'area': [], 'cov': [], 'snr': []} for nu in N_UCB_GRID}

    for nu in N_UCB_GRID:
        for i in idxs:
            e = ds[i]
            t = e['target'].numpy()
            base = e['meta'].unsqueeze(0).to(device)
            # clean signal + injected UCB foreground (this n_ucb) + instrument noise
            fd0 = e['fd'].unsqueeze(0).to(device).clone()
            g = torch.Generator(device=device).manual_seed(int(i) * 17 + nu)
            n_ucb = torch.tensor([nu], device=device)
            inject_ucb_foreground(fd0, n_ucb, g_bin, band, rng=g)
            fd0[:, :, :4] += torch.randn(fd0[:, :, :4].shape, generator=g,
                                         device=device) * sigma_d[None, :, None]
            ucb_meta = torch.tensor([[nu / UCB_N_MAX_COND]], device=device, dtype=torch.float32)
            fd = (fd0 - fmean) / fstd

            sb, lam, s = 0.5, 0.5, SKY_SIGMA_MAX
            with torch.no_grad():
                for r in range(args.rounds):
                    pose = sky_pose_meta(torch.tensor([[sb, lam]], device=device, dtype=torch.float32),
                                         torch.tensor([s], device=device, dtype=torch.float32),
                                         gen=torch.Generator(device=device).manual_seed(0))
                    meta = torch.cat([base, ucb_meta, pose], dim=-1)
                    samp = m.sample(fd, args.n_post, meta=meta).cpu().numpy().squeeze()
                    samp = np.clip(samp, 1e-4, 1 - 1e-4)
                    sb = float(np.median(samp[:, 6])); lam = circ_median(samp[:, 7])
                    s = float(np.clip(max(np.std(samp[:, 6]), np.std(samp[:, 7])),
                                      SKY_SIGMA_MIN, SKY_SIGMA_MAX))
            res[nu]['area'].append(sky_area_90(samp[:, 7] * 2 * np.pi, samp[:, 6] * 2 - 1))
            res[nu]['cov'].append(0.05 < (samp[:, 7] <= t[7]).mean() < 0.95)
            res[nu]['snr'].append(float(e['snr']))

    print(f"\nRobustness sweep ({len(idxs)} events, {args.rounds} GNPE rounds):")
    print(f"{'n_ucb':>6} {'med area':>9} {'lam cov':>8} {'hiSNR area':>11}")
    for nu in N_UCB_GRID:
        a = np.array(res[nu]['area']); snr = np.array(res[nu]['snr'])
        hi = snr > 200
        print(f"{nu:>6} {np.median(a):9.0f} {100*np.mean(res[nu]['cov']):7.0f}% "
              f"{np.median(a[hi]) if hi.sum() else float('nan'):11.0f}")
    np.savez(product_out('robust_sweep.npz'), **{f'area_{nu}': res[nu]['area'] for nu in N_UCB_GRID},
             **{f'snr_{nu}': res[nu]['snr'] for nu in N_UCB_GRID})


if __name__ == '__main__':
    main()
