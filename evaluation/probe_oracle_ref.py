"""
TRAINING-FREE PROBE: does an accurate intrinsic reference sharpen the sky?

Hypothesis (memory sky-localization-diagnosis line 46): the sky-GNPE sky floors
at ~1700 deg^2 because the INTRINSICS (Mc, t, phase) are only conditioned via a
fixed stage-1 reference jittered at the kernel scale (~0.1-0.3% Mc), and a 0.1%
Mc / 30s t error shifts the sharp coherent sky peak ~2.4 deg off truth. The
sky-GNPE iterates the SKY pose but never refines the intrinsics.

The MBHB flow is conditioned on the reference (ref_meta_vec FiLM) AND was
trained over a kernel-scale ladder (logU[0.02,1] + 15% s=4). So we can feed it an
ORACLE reference (ref = exact truth, small kernel scale) and ask: does the sky
tighten vs a realistic jittered reference?

  oracle sky << jittered sky  -> coupling IS the bottleneck -> ridge-pose GNPE worth training
  oracle sky ~= jittered sky  -> encoder can't exploit accurate intrinsics -> pivot to scale-up

Run: python evaluation/probe_oracle_ref.py --n-events 4
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
from pe_smbhb.train_flow import (build_model, sky_pose_meta, ref_meta_vec,
                              sky_area_90, SKY_SIGMA_MIN, SKY_SIGMA_MAX)

SANGRIA = dict(m1=4956676.2876, m2=4067166.60352, chi1=-0.523732715,
               chi2=-0.117412144, dist_Mpc=61097.116076, iota=1.420048341,
               beta=-1.081082148, lam=4.052962883, psi=1.22844350008,
               phi0=0.6417162631)


from pe_smbhb.inference import circ_median     # noqa: E402,F401

def build_fd(gen, dA, dE, ref, fmean, fstd, device):
    """Heterodyned binned features for a given reference (intrinsics + fiducial ext)."""
    pr = np.array([[ref['Mc'], ref['q'], ref['c1'], ref['c2'], FIDUCIAL_EXT['logD'],
                    ref['t_ref_hr'], FIDUCIAL_EXT['ci'], FIDUCIAL_EXT['sb'],
                    FIDUCIAL_EXT['lam'], FIDUCIAL_EXT['psi'], FIDUCIAL_EXT['phi0']]])
    Ar, _ = gen.gen_wf(pr); Ar = Ar[0]; mag = cp.abs(Ar)
    c = cp.where(mag > 0, cp.conj(Ar) / cp.where(mag > 0, mag, 1.0), 0.0)
    rAb = gen.bin_mean((dA * c)[None, :])[0]; rEb = gen.bin_mean((dE * c)[None, :])[0]
    href = gen.bin_mean((mag / gen.sqrt_psd_g).astype(cp.complex128)[None, :])[0].real
    feats = cp.stack([rAb.real, rAb.imag, rEb.real, rEb.imag, cp.log1p(cp.maximum(href, 0))], -1)
    fd = torch.as_tensor(cp.asnumpy(feats), dtype=torch.float32, device=device).unsqueeze(0)
    return (fd - fmean) / fstd


def run_gnpe(m, fd, base, cfg, n_post, rounds, inflate, device):
    """Iterative sky-GNPE; return samples (n_post, 11) and 90% sky area."""
    sb_s, lam_s, sig = 0.5, 0.5, SKY_SIGMA_MAX
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
    a90 = sky_area_90(samp[:4000, 7] * 2 * np.pi, samp[:4000, 6] * 2 - 1)
    return samp, a90


def make_ref(truth, t_ref_hr, kbase, mode, s_oracle, rng):
    """Reference intrinsics.
      mode='jitter'    -> truth + kernel jitter at s=1 (realistic, matches eval_sangria).
      mode='oracle'    -> all intrinsics = truth exactly, small kernel scale s_oracle.
      mode='oracle_mct'-> Mc & t exact (small s), but q,c1,c2 jittered at s=1.
    Tests whether accurate {Mc,t} alone drives the sky tightening (=> pose only those,
    avoiding spin overconfidence) or whether the spins matter too."""
    Mc, q = truth['Mc'], truth['q']
    k = kbase
    if mode == 'oracle':
        return dict(Mc=Mc, q=q, c1=truth['c1'], c2=truth['c2'], t_ref_hr=t_ref_hr,
                    kernel_scale=s_oracle, s_phase=s_oracle, kernel_type=0)
    jq = max(q * np.exp(rng.randn() * k['log_q']), 1.0001)
    jc1 = float(np.clip(truth['c1'] + rng.randn() * k['chi'], -0.99, 0.99))
    jc2 = float(np.clip(truth['c2'] + rng.randn() * k['chi'], -0.99, 0.99))
    if mode == 'oracle_mct':
        return dict(Mc=Mc, q=jq, c1=jc1, c2=jc2, t_ref_hr=t_ref_hr,
                    kernel_scale=s_oracle, s_phase=s_oracle, kernel_type=0)
    return dict(
        Mc=Mc * (1 + rng.randn() * k['frac_Mc']),
        q=jq, c1=jc1, c2=jc2,
        t_ref_hr=t_ref_hr + rng.randn() * max(k['t_floor_sec'],
                 k['t_frac_of_tpre'] * truth['t_pre'] * 3600) / 3600,
        kernel_scale=1.0, s_phase=1.0, kernel_type=0)


def make_ref_err(truth, t_ref_hr, kbase, frac_mc, t_err_sec, s_decl, rng):
    """Reference with a CONTROLLED Mc,t error (q,c1,c2 jittered since spins don't
    matter per oracle_mct). frac_mc = fractional Mc error, t_err_sec = t error in s.
    Declared kernel scale s_decl (the model trusts it). The sign of the error is
    randomized per call so we sample both ridge directions."""
    k = kbase
    sgn = 1.0 if rng.rand() < 0.5 else -1.0
    return dict(
        Mc=truth['Mc'] * (1 + sgn * frac_mc),
        q=max(truth['q'] * np.exp(rng.randn() * k['log_q']), 1.0001),
        c1=float(np.clip(truth['c1'] + rng.randn() * k['chi'], -0.99, 0.99)),
        c2=float(np.clip(truth['c2'] + rng.randn() * k['chi'], -0.99, 0.99)),
        t_ref_hr=t_ref_hr + sgn * t_err_sec / 3600.0,
        kernel_scale=s_decl, s_phase=s_decl, kernel_type=0)


def event_data(gen, p_true, ref_seed):
    A, E = gen.gen_wf(p_true)
    rng = cp.random.RandomState(ref_seed)
    dA = A[0] / gen.sqrt_psd_g + (rng.standard_normal(gen.n_bins) + 1j * rng.standard_normal(gen.n_bins)) * gen.sigma0
    dE = E[0] / gen.sqrt_psd_g + (rng.standard_normal(gen.n_bins) + 1j * rng.standard_normal(gen.n_bins)) * gen.sigma0
    snr = float(cp.sqrt(4 * gen.df * cp.sum((cp.abs(A[0])**2 + cp.abs(E[0])**2) / gen.psd_g)))
    return dA, dE, snr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=MBHB_FLOW)
    ap.add_argument('--norm', default=MBHB_NORM)
    ap.add_argument('--data', default=None)
    ap.add_argument('--t-pre', type=float, default=24.0)
    ap.add_argument('--rounds', type=int, default=4)
    ap.add_argument('--inflate', type=float, default=1.3)
    ap.add_argument('--n-post', type=int, default=20000)
    ap.add_argument('--s-oracle', type=float, default=0.1)
    ap.add_argument('--n-events', type=int, default=4, help='random events besides Sangria')
    ap.add_argument('--sweep', action='store_true', help='sweep Mc,t reference error vs sky area')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    device = torch.device(args.device)

    ck = torch.load(args.model, map_location=device, weights_only=False)
    cfg = ck['config']
    m = build_model(cfg).to(device); m.load_state_dict(ck['model_state_dict']); m.eval()
    nm = torch.load(args.norm, map_location=device, weights_only=False)
    fmean, fstd = nm['fd_mean'].to(device), nm['fd_std'].to(device)
    kbase = load_kernel(args.data)
    print(f"model {Path(args.model).name} ep{ck['epoch']} val_sky {ck.get('val_sky', float('nan')):.3f}; "
          f"oracle s={args.s_oracle}, jitter s=1, rounds={args.rounds}")

    gen = HetDataGenerator(output_dir=None)
    np.random.seed(123)
    for _ in range(2):
        Pw, _ = gen.draw_priors(64); gen.gen_wf(Pw)
    t_ref_hr = (gen.t_obs - args.t_pre * 3600) / 3600

    # --- build event list: Sangria + random prior draws (decent SNR) ---
    s_m1, s_m2 = SANGRIA['m1'], SANGRIA['m2']; s_eta = s_m1 * s_m2 / (s_m1 + s_m2)**2
    s_Mc = (s_m1 + s_m2) * s_eta**0.6
    events = [dict(name='Sangria', Mc=s_Mc, q=s_m1 / s_m2, c1=SANGRIA['chi1'], c2=SANGRIA['chi2'],
                   logD=np.log10(SANGRIA['dist_Mpc']), ci=np.cos(SANGRIA['iota']),
                   sb=np.sin(SANGRIA['beta']), lam=SANGRIA['lam'], psi=SANGRIA['psi'],
                   phi0=SANGRIA['phi0'], t_pre=args.t_pre)]
    rprng = np.random.default_rng(7)
    while len(events) < args.n_events + 1:
        P, _ = gen.draw_priors(64)
        for row in P:
            Mc, q, c1, c2, logD, t_ref, ci, sb, lam, psi, phi0 = row
            t_pre = (gen.t_obs / 3600) - t_ref
            if not (5 < t_pre < 200):     # pick events in the few-hr..week pre-merger band
                continue
            events.append(dict(name=f'rand{len(events)}', Mc=Mc, q=q, c1=c1, c2=c2, logD=logD,
                               ci=ci, sb=sb, lam=lam, psi=psi, phi0=phi0, t_pre=t_pre))
            break
        if len(events) >= args.n_events + 1:
            break

    if args.sweep:
        # (frac_Mc error, t error [s]) pairs spanning oracle -> realistic.
        # 2.4 deg sky shift per 0.1% Mc (memory line 46) -> scan around there.
        LEVELS = [(0.0, 0.0), (3e-4, 1.0), (1e-3, 5.0), (3e-3, 30.0), (1e-2, 100.0)]
        s_decl = 0.1
        print(f"\nMc,t reference-error sweep (q,c1,c2 jittered; declared s={s_decl}); "
              f"median 90% sky area [deg^2] over {len(events)} events\n")
        print(f"{'dMc/Mc':>8} {'dt[s]':>7} | " + " ".join(f"{e['name']:>7}" for e in events) + " |   median")
        for frac_mc, t_err in LEVELS:
            areas = []
            for ev in events:
                t_ref = (gen.t_obs - ev['t_pre'] * 3600) / 3600
                p_true = np.array([[ev['Mc'], ev['q'], ev['c1'], ev['c2'], ev['logD'], t_ref,
                                    ev['ci'], ev['sb'], ev['lam'], ev['psi'], ev['phi0']]])
                dA, dE, snr = event_data(gen, p_true, ref_seed=0)
                truth = dict(Mc=ev['Mc'], q=ev['q'], c1=ev['c1'], c2=ev['c2'], t_pre=ev['t_pre'])
                ref = make_ref_err(truth, t_ref, kbase, frac_mc, t_err, s_decl, np.random.RandomState(1))
                fd = build_fd(gen, dA, dE, ref, fmean, fstd, device)
                base = torch.from_numpy(ref_meta_vec(ref)).unsqueeze(0).to(device).float()
                if cfg.get('ucb_cond', False):
                    base = torch.cat([base, torch.zeros(1, 1, device=device)], dim=-1)
                _, a90 = run_gnpe(m, fd, base, cfg, args.n_post, args.rounds, args.inflate, device)
                areas.append(a90)
            tag = 'oracle' if frac_mc == 0 else ''
            print(f"{frac_mc:>8.0e} {t_err:>7.1f} | " + " ".join(f"{a:>7.0f}" for a in areas)
                  + f" | {np.median(areas):>8.0f} {tag}")
        print("\nReads off how accurate Mc,t must be to realize the sky gain. The model's own "
              "intrinsic posterior floors at ~0.1-0.3% Mc -> compare that row to oracle.")
        return

    MODES = ('jitter', 'oracle_mct', 'oracle')
    print(f"\n{'event':>8} {'SNR':>6} | {'jitter':>9} | {'oracle_Mct':>10} | {'oracle_all':>10}")
    rows = []
    for ev in events:
        t_ref = (gen.t_obs - ev['t_pre'] * 3600) / 3600
        p_true = np.array([[ev['Mc'], ev['q'], ev['c1'], ev['c2'], ev['logD'], t_ref,
                            ev['ci'], ev['sb'], ev['lam'], ev['psi'], ev['phi0']]])
        dA, dE, snr = event_data(gen, p_true, ref_seed=0)
        truth = dict(Mc=ev['Mc'], q=ev['q'], c1=ev['c1'], c2=ev['c2'], t_pre=ev['t_pre'])

        res = {}
        for mode in MODES:
            ref = make_ref(truth, t_ref, kbase, mode, args.s_oracle, np.random.RandomState(1))
            fd = build_fd(gen, dA, dE, ref, fmean, fstd, device)
            base = torch.from_numpy(ref_meta_vec(ref)).unsqueeze(0).to(device).float()
            if cfg.get('ucb_cond', False):
                base = torch.cat([base, torch.zeros(1, 1, device=device)], dim=-1)
            _, a90 = run_gnpe(m, fd, base, cfg, args.n_post, args.rounds, args.inflate, device)
            res[mode] = a90
        rows.append((ev['name'], snr, res['jitter'], res['oracle_mct'], res['oracle']))
        print(f"{ev['name']:>8} {snr:>6.0f} | {res['jitter']:>9.0f} | {res['oracle_mct']:>10.0f} "
              f"| {res['oracle']:>10.0f}")

    j = np.array([r[2] for r in rows]); omct = np.array([r[3] for r in rows]); o = np.array([r[4] for r in rows])
    print(f"\nmedian: jitter {np.median(j):.0f}  ->  oracle_Mct {np.median(omct):.0f}  "
          f"->  oracle_all {np.median(o):.0f} deg^2")
    print(f"  Mc,t alone capture {100*(np.median(j)-np.median(omct))/max(np.median(j)-np.median(o),1):.0f}% "
          f"of the oracle gain (rest needs accurate spins/q)")
    print("\nVERDICT:", "ORACLE MUCH TIGHTER -> coupling is the bottleneck; ridge-pose GNPE worth training."
          if np.median(o) < 0.5 * np.median(j) else
          "oracle ~= jitter -> encoder cannot exploit accurate intrinsics; pivot to scale-up.")


if __name__ == '__main__':
    main()
