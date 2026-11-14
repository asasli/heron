"""
Iterative sky-GNPE inference + area measurement.

Round 0: seedless (sky_sigma = SKY_SIGMA_MAX, pose at prior centre) -> broad
         sky posterior. Round r: seed = posterior circular-median (sin_beta,
         lambda), sky_sigma = max(posterior std, floor) -> sharper. Gibbs.
Reports median 90% sky area per round, by SNR, + coverage.

Run: python evaluation/eval_skygnpe.py --model weights/mbhb_flow.pt --rounds 4
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
                              SKY_SIGMA_MIN, SKY_SIGMA_MAX)


def circ_median(lam01):
    ang = lam01 * 2 * np.pi
    return (np.angle(np.mean(np.exp(1j * ang))) % (2 * np.pi)) / (2 * np.pi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=MBHB_FLOW)
    ap.add_argument('--norm', default=MBHB_NORM)
    ap.add_argument('--data', default=MBHB_DATA)
    ap.add_argument('--rounds', type=int, default=4)
    ap.add_argument('--n-post', type=int, default=3000)
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    torch.manual_seed(0)              # deterministic flow draws
    device = torch.device(args.device)

    ck = torch.load(args.model, map_location=device, weights_only=False)
    cfg = ck['config']
    m = build_model(cfg).to(device)
    m.load_state_dict(ck['model_state_dict']); m.eval()
    print(f"model epoch {ck['epoch']} val_sky {ck.get('val_sky'):.3f} n_meta {cfg['n_meta']}")
    nm = torch.load(args.norm, map_location=device, weights_only=False)
    fmean, fstd = nm['fd_mean'].to(device), nm['fd_std'].to(device)
    ds = HetDataset(args.data, 'test', residual=True)
    ucb_cond = ck['config'].get('ucb_cond', False)

    idxs = list(range(0, len(ds), args.stride))
    R = args.rounds
    areas = np.zeros((len(idxs), R)); snr = np.zeros(len(idxs))
    cov = np.zeros((len(idxs), R), bool)
    with torch.no_grad():
        for n, i in enumerate(idxs):
            e = ds[i]
            fd = ((e['fd'].unsqueeze(0).to(device)) - fmean) / fstd
            base = e['meta'].unsqueeze(0).to(device)
            if ucb_cond:  # clean eval: n_ucb=0 conditioning dim
                base = torch.cat([base, torch.zeros(1, 1, device=device)], dim=-1)
            t = e['target'].numpy()
            sb_seed, lam_seed, sig = 0.5, 0.5, SKY_SIGMA_MAX
            for r in range(R):
                pose = sky_pose_meta(torch.tensor([[sb_seed, lam_seed]], device=device, dtype=torch.float32),
                                     torch.tensor([sig], device=device, dtype=torch.float32),
                                     gen=torch.Generator(device=device).manual_seed(0))
                meta = torch.cat([base, pose], dim=-1)
                s = m.sample(fd, args.n_post, meta=meta).cpu().numpy().squeeze()
                s = np.clip(s, 1e-4, 1 - 1e-4)
                sb_s, lam_s = s[:, 6], s[:, 7]
                areas[n, r] = sky_area_90(lam_s * 2 * np.pi, sb_s * 2 - 1)
                cov[n, r] = 0.05 < (s[:, 7] <= t[7]).mean() < 0.95
                # next seed
                lam_seed = circ_median(lam_s)
                sb_seed = float(np.median(sb_s))
                sig = float(np.clip(max(np.std(sb_s), np.std(lam_s)),
                                    SKY_SIGMA_MIN, SKY_SIGMA_MAX))
            snr[n] = float(e['snr'])

    print(f"\nIterative sky-GNPE ({len(idxs)} events):")
    for r in range(R):
        print(f"  round {r}: median {np.median(areas[:, r]):6.0f} deg^2 | "
              f"lambda coverage {100*cov[:, r].mean():.0f}%")
    print("final round by SNR:")
    for lo, hi in [(10, 50), (50, 200), (200, 1000), (1000, 1e9)]:
        mk = (snr >= lo) & (snr < hi)
        if mk.sum() > 3:
            print(f"  SNR[{lo:5d},{hi:7.0f}): {np.median(areas[mk, -1]):6.0f} deg^2 "
                  f"(n={mk.sum()}, best {areas[mk, -1].min():.0f})")
    np.savez(product_out('skygnpe_eval.npz'), areas=areas, snr=snr, cov=cov)


if __name__ == '__main__':
    main()
