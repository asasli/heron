"""
Inference-side reference refinement (no retraining). The oracle-error sweep
(scripts/probe_oracle_ref.py --sweep) showed the sky tightens ~5-9x once the
conditioning reference is (a) accurate to ~0.1-0.3% Mc / ~5-30 s t AND (b)
DECLARED tight (small kernel scale s) so the model stops hedging broad.

The model's OWN round-0 intrinsic posterior median is the cheap accurate-enough
estimate. So: round-0 (loose reference) -> take intrinsic medians -> rebuild the
heterodyne with that refined reference, DECLARED tight -> re-run sky-GNPE.

This measures, per event:
  - the model's actual intrinsic median accuracy (dMc/Mc, dt) — the thing the
    sweep assumed;
  - baseline sky area (loose reference) vs refined sky area;
  - CALIBRATION: is truth inside the 90% CI for sky (sin_beta, lambda) and
    intrinsics (Mc, t)? Declaring tight risks overconfidence -> must check.

Run: python evaluation/refine_infer.py --n-events 4 --s-tight 0.3
"""
import sys, argparse, json
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from heron.artifacts import (MBHB_FLOW, MBHB_NORM, MBHB_RECAL, UCB_FLOW, UCB_RESPONSE,
                             MBHB_DATA, UCB_DATA, load_kernel, product_out,
                             product_in, figure_out)
import cupy as cp
from pe_smbhb.data_mbhb import HetDataGenerator, FIDUCIAL_EXT
from pe_smbhb.train_flow import (build_model, sky_pose_meta, ref_meta_vec, residual_to_physical,
                              sky_area_90, SKY_SIGMA_MIN, SKY_SIGMA_MAX)
from pe_smbhb.networks import PARAM_RANGES
from evaluation.probe_oracle_ref import (SANGRIA, circ_median, build_fd, make_ref, event_data)

T_OBS_HR = 2.6298e6 / 3600.0


def sample_gnpe(m, fd, base, n_post, rounds, inflate, device, seed_sky=None, sig0=SKY_SIGMA_MAX):
    """Iterative sky-GNPE; returns final samples (n_post,11)."""
    if seed_sky is None:
        sb_s, lam_s, sig = 0.5, 0.5, sig0
    else:
        sb_s, lam_s, sig = seed_sky[0], seed_sky[1], sig0
    with torch.no_grad():
        for r in range(rounds):
            pose = sky_pose_meta(torch.tensor([[sb_s, lam_s]], device=device, dtype=torch.float32),
                                 torch.tensor([sig], device=device, dtype=torch.float32),
                                 gen=torch.Generator(device=device).manual_seed(0))
            meta = torch.cat([base, pose], dim=-1)
            samp = m.sample(fd, n_post, meta=meta).cpu().numpy().squeeze()
            samp = np.clip(samp, 1e-4, 1 - 1e-4)
            sb_s = float(np.median(samp[:, 6])); lam_s = circ_median(samp[:, 7])
            sig = float(np.clip(max(np.std(samp[:, 6]), np.std(samp[:, 7])) * inflate,
                                SKY_SIGMA_MIN, SKY_SIGMA_MAX))
    return samp


def mk_base(ref, cfg, device):
    base = torch.from_numpy(ref_meta_vec(ref)).unsqueeze(0).to(device).float()
    if cfg.get('ucb_cond', False):
        base = torch.cat([base, torch.zeros(1, 1, device=device)], dim=-1)
    return base


def covered(samp_col, truth, lo=5, hi=95):
    a, b = np.percentile(samp_col, [lo, hi])
    return a <= truth <= b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=MBHB_FLOW)
    ap.add_argument('--norm', default=MBHB_NORM)
    ap.add_argument('--data', default=None)
    ap.add_argument('--t-pre', type=float, default=24.0)
    ap.add_argument('--rounds', type=int, default=4)
    ap.add_argument('--inflate', type=float, default=1.3)
    ap.add_argument('--n-post', type=int, default=20000)
    ap.add_argument('--s-tight', type=float, default=0.3, help='declared kernel scale for refined ref')
    ap.add_argument('--refine-iters', type=int, default=2, help='reference-refinement Gibbs passes')
    ap.add_argument('--n-events', type=int, default=4)
    ap.add_argument('--noise-seed', type=int, default=0)
    ap.add_argument('--mc-bias-corr', type=float, default=0.0, help='empirical median recalibration: Mc_med *= (1+corr)')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    device = torch.device(args.device)

    ck = torch.load(args.model, map_location=device, weights_only=False)
    cfg = ck['config']
    m = build_model(cfg).to(device); m.load_state_dict(ck['model_state_dict']); m.eval()
    res_p = cfg.get('residual_params') or None
    wide_spin = res_p is not None and 'c1' not in (res_p or [])  # v15: abs spins -> wide refs
    nm = torch.load(args.norm, map_location=device, weights_only=False)
    fmean, fstd = nm['fd_mean'].to(device), nm['fd_std'].to(device)
    kbase = load_kernel(args.data)
    print(f"model {Path(args.model).name} ep{ck['epoch']}; s_tight={args.s_tight}, "
          f"refine_iters={args.refine_iters}, rounds={args.rounds}")

    gen = HetDataGenerator(output_dir=None)
    np.random.seed(123)
    for _ in range(2):
        Pw, _ = gen.draw_priors(64); gen.gen_wf(Pw)

    s_m1, s_m2 = SANGRIA['m1'], SANGRIA['m2']; s_eta = s_m1 * s_m2 / (s_m1 + s_m2)**2
    s_Mc = (s_m1 + s_m2) * s_eta**0.6
    events = [dict(name='Sangria', Mc=s_Mc, q=s_m1 / s_m2, c1=SANGRIA['chi1'], c2=SANGRIA['chi2'],
                   logD=np.log10(SANGRIA['dist_Mpc']), ci=np.cos(SANGRIA['iota']),
                   sb=np.sin(SANGRIA['beta']), lam=SANGRIA['lam'], psi=SANGRIA['psi'],
                   phi0=SANGRIA['phi0'], t_pre=args.t_pre)]
    while len(events) < args.n_events + 1:
        P, _ = gen.draw_priors(64)
        for row in P:
            Mc, q, c1, c2, logD, t_ref, ci, sb, lam, psi, phi0 = row
            t_pre = (gen.t_obs / 3600) - t_ref
            if not (5 < t_pre < 200):
                continue
            events.append(dict(name=f'rand{len(events)}', Mc=Mc, q=q, c1=c1, c2=c2, logD=logD,
                               ci=ci, sb=sb, lam=lam, psi=psi, phi0=phi0, t_pre=t_pre))
            break

    hdr = (f"{'event':>8} {'SNR':>5} | {'dMc/Mc':>8} {'dt[s]':>7} | "
           f"{'base sky':>8} {'ref sky':>8} | {'covSky':>6} {'covMc':>5} {'covT':>5}")
    print("\n" + hdr)
    rows = []
    for ev in events:
        t_ref_hr = (gen.t_obs - ev['t_pre'] * 3600) / 3600
        p_true = np.array([[ev['Mc'], ev['q'], ev['c1'], ev['c2'], ev['logD'], t_ref_hr,
                            ev['ci'], ev['sb'], ev['lam'], ev['psi'], ev['phi0']]])
        dA, dE, snr = event_data(gen, p_true, ref_seed=args.noise_seed)
        truth = dict(Mc=ev['Mc'], q=ev['q'], c1=ev['c1'], c2=ev['c2'], t_pre=ev['t_pre'])
        t_pre_true = ev['t_pre']

        # --- round-0 with the realistic loose reference ---
        ref0 = make_ref(truth, t_ref_hr, kbase, 'jitter', 0.1, np.random.RandomState(1))
        if wide_spin:
            rr = np.random.RandomState(2)
            ref0['c1'] = float(rr.uniform(-0.99, 0.99)); ref0['c2'] = float(rr.uniform(-0.99, 0.99))
        fd0 = build_fd(gen, dA, dE, ref0, fmean, fstd, device)
        base0 = mk_base(ref0, cfg, device)
        samp0 = sample_gnpe(m, fd0, base0, args.n_post, args.rounds, args.inflate, device)
        a_base = sky_area_90(samp0[:4000, 7] * 2 * np.pi, samp0[:4000, 6] * 2 - 1)

        # --- reference refinement: model intrinsic medians -> tight-declared ref ---
        ref = ref0
        samp = samp0
        for it in range(args.refine_iters):
            phys = residual_to_physical(samp, ref, kbase, params=res_p)
            Mc_med = float(np.median(phys['Mc'])) * (1.0 + args.mc_bias_corr)
            q_med = float(np.median(phys['q']))
            c1_med = float(np.median(phys['c1'])); c2_med = float(np.median(phys['c2']))
            tpre_med = float(np.median(phys['t_pre_hours']))
            rr = np.random.RandomState(100 + it)
            ref = dict(Mc=Mc_med, q=max(q_med, 1.0001),
                       c1=(float(rr.uniform(-0.99, 0.99)) if wide_spin
                           else float(np.clip(c1_med, -0.99, 0.99))),
                       c2=(float(rr.uniform(-0.99, 0.99)) if wide_spin
                           else float(np.clip(c2_med, -0.99, 0.99))),
                       t_ref_hr=(gen.t_obs / 3600) - tpre_med,
                       kernel_scale=args.s_tight, s_phase=args.s_tight, kernel_type=0)
            fd = build_fd(gen, dA, dE, ref, fmean, fstd, device)
            base = mk_base(ref, cfg, device)
            samp = sample_gnpe(m, fd, base, args.n_post, args.rounds, args.inflate, device)

        phys = residual_to_physical(samp, ref, kbase, params=res_p)
        dMc = float(np.median(phys['Mc'])) / ev['Mc'] - 1.0
        dt = (float(np.median(phys['t_pre_hours'])) - t_pre_true) * 3600.0
        a_ref = sky_area_90(samp[:4000, 7] * 2 * np.pi, samp[:4000, 6] * 2 - 1)

        # calibration (truth-in-90% CI), refined posterior
        cov_sb = covered(samp[:, 6] * 2 - 1, ev['sb']); cov_lam = covered(samp[:, 7] * 2 * np.pi, ev['lam'])
        cov_sky = cov_sb and cov_lam
        cov_mc = covered(phys['Mc'], ev['Mc']); cov_t = covered(phys['t_pre_hours'], t_pre_true)
        rows.append((ev['name'], snr, dMc, dt, a_base, a_ref, cov_sky, cov_mc, cov_t))
        print(f"{ev['name']:>8} {snr:>5.0f} | {dMc:>+8.1e} {dt:>+7.1f} | "
              f"{a_base:>8.0f} {a_ref:>8.0f} | {str(cov_sky):>6} {str(cov_mc):>5} {str(cov_t):>5}")

    ab = np.array([r[4] for r in rows]); ar = np.array([r[5] for r in rows])
    print(f"\nmedian sky: baseline {np.median(ab):.0f} -> refined {np.median(ar):.0f} deg^2 "
          f"({np.median(ab)/max(np.median(ar),1):.1f}x)")
    print(f"coverage(refined): sky {np.mean([r[6] for r in rows])*100:.0f}%  "
          f"Mc {np.mean([r[7] for r in rows])*100:.0f}%  t {np.mean([r[8] for r in rows])*100:.0f}%  "
          f"(of {len(rows)} events; want ~90%)")


if __name__ == '__main__':
    main()
