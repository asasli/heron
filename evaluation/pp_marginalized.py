"""
Calibration of the alert posterior over a population (Fig. 3, Extended Data).

Fresh injections are drawn from the prior with the GPU waveform (bbhx), given
an independent noise realization and a resolved-UCB foreground (`--n-ucb`
lines injected with the calibrated per-bin response, weights/ucb_response.npz),
and analysed exactly as in the pipeline: one heterodyne reference
(`--k-ref 1`, drawn at kernel scale `--ref-scale`), sky-GNPE with K_pose = 8
pose draws in the last round. For every event the script records the
percentile rank of the truth in each marginal posterior and the credibility
level at which the true sky position enters the two-dimensional credible
region. Output: a rank file (results/products/pp_ranks.npz by default) and the
raw probability-probability plot. evaluation/fit_recalibration.py turns the
rank files into the calibration map and the published Fig. 3a.

Published configuration (K_ref = 1, prior-drawn reference spins, n_UCB = 20):
    # held-out test set of Fig. 3: 500 injections, reference kernel scale 1.0
    python evaluation/pp_marginalized.py --n-events 500 --ref-scale 1.0
    # independent calibration set behind the released map: 3000 injections
    python evaluation/pp_marginalized.py --n-events 3000 --ref-scale 1.0 \
        --seed-base 10000 --out-ranks results/products/pp_ranks_cal.npz
    # the sky-coverage run quoted in the main text (55.2% / 89.8%): the same
    # 500 events at the pipeline's operating scale
    python evaluation/pp_marginalized.py --n-events 500 --ref-scale 0.45 \
        --out-ranks results/products/pp_ranks_skycov.npz
The calibration map is specific to the reference scale it was fitted at
(docs/REPRODUCE.md, "Reference scale").

Reference pooling (`--k-ref 16`) and self-refinement (`--refine`) are the two
inference-side schemes shown not to remove the reference-echo floor
(Supplementary Note "Stage-2 amortized inference and the amortization floor").

The events, noise and references are seeded, and so are the flow draws: a
rerun on the same hardware and software reproduces the rank file. The
published files were produced before the flow draws were seeded, so a rerun
is statistically equivalent to them, not identical (docs/REPRODUCE.md).
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
from pe_smbhb.data_mbhb import HetDataGenerator, FIDUCIAL_EXT
from pe_smbhb.train_flow import (build_model, sky_pose_meta, ref_meta_vec, residual_to_physical,
                              inject_ucb_foreground, SKY_SIGMA_MIN, SKY_SIGMA_MAX,
                              UCB_N_MAX_COND)
from pe_smbhb.networks import PARAM_NAMES, PARAM_RANGES
from evaluation.plotstyle import paper_style as _paper_style

from pe_smbhb.inference import draw_ref, het_features, gnpe_marg     # noqa: E402,F401

K_REF, K_POSE = 1, 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=MBHB_FLOW)
    ap.add_argument('--norm', default=MBHB_NORM)
    ap.add_argument('--data', default=None)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--inflate', type=float, default=1.6)
    ap.add_argument('--n-events', type=int, default=300)
    ap.add_argument('--n-per', type=int, default=400, help='samples per (ref,pose) member')
    ap.add_argument('--n-ucb', type=int, default=20)
    ap.add_argument('--k-ref', type=int, default=1,
                    help='heterodyne references pooled per event (published: 1)')
    ap.add_argument('--spin-marg', action='store_true', help='marginalize ONLY spin reference; share Mc/q/t (K_REF>1)')
    ap.add_argument('--k-pose', type=int, default=8)
    ap.add_argument('--wide-spin-ref', action='store_true', default=True,
                    help='reference spins drawn from the prior (default; the released model)')
    ap.add_argument('--truth-centred-spin-ref', dest='wide_spin_ref', action='store_false',
                    help='reference spins jittered around the truth (models trained without wide-spin references)')
    ap.add_argument('--refine', action='store_true',
                    help='self-refined reference: round-0 k=1 medians -> re-heterodyne (use with --k-ref 1)')
    ap.add_argument('--refine-scale', type=float, default=0.45, help='declared kernel scale of the refined ref')
    ap.add_argument('--ref-scale', type=float, default=1.0,
                    help='kernel scale of the heterodyne reference draw. 1.0: the published '
                         'calibration and test sets (Fig. 3) and the released calibration map; '
                         '0.45: the operating point of the pipeline alert and of the published '
                         'sky-coverage run')
    ap.add_argument('--seed-base', type=int, default=0, help='offset all RNG seeds (independent event sets)')
    ap.add_argument('--out-ranks', default=None,
                    help='where to save the percentile ranks (default results/products/pp_ranks.npz)')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    args.out_ranks = args.out_ranks or product_out('pp_ranks.npz')
    SB = args.seed_base
    device = torch.device(args.device)
    global K_REF, K_POSE
    K_REF, K_POSE = args.k_ref, args.k_pose

    ck = torch.load(args.model, map_location=device, weights_only=False)
    cfg = ck['config']
    m = build_model(cfg).to(device); m.load_state_dict(ck['model_state_dict']); m.eval()
    res_p = cfg.get('residual_params') or None  # v14abs: only Mc,t residual; q/chi absolute
    nm = torch.load(args.norm, map_location=device, weights_only=False)
    fmean, fstd = nm['fd_mean'].to(device), nm['fd_std'].to(device)
    kbase = load_kernel(args.data)
    print(f"model ep{ck['epoch']} val_sky {ck.get('val_sky', float('nan')):.3f} | "
          f"K_REF={K_REF} K_POSE={K_POSE} rounds={args.rounds} n_ucb={args.n_ucb}")

    if args.n_ucb > 0:
        resp = np.load(UCB_RESPONSE)
        g_bin = torch.from_numpy(resp['g_bin'].astype(np.float32)).to(device)
        band = torch.from_numpy(np.where(resp['g_bin'] > 0)[0].astype(np.int64)).to(device)

    gen = HetDataGenerator(output_dir=None)
    np.random.seed(2000 + SB)
    for _ in range(2):
        Pw, _ = gen.draw_priors(64); gen.gen_wf(Pw)

    pct = {n: [] for n in PARAM_NAMES}
    sky_cred = []          # 2D sky HPD credibility of the truth (joint sky validation)
    done = 0
    while done < args.n_events:
        P, t_pre = gen.draw_priors(64)             # prior-diverse batch (bbhx-safe)
        A, E = gen.gen_wf(P)
        for b in range(64):
            if done >= args.n_events:
                break
            if not (0.05 < t_pre[b] < 600):
                continue
            rng = cp.random.RandomState(1000 + SB + done)
            dA = A[b]/gen.sqrt_psd_g + (rng.standard_normal(gen.n_bins)+1j*rng.standard_normal(gen.n_bins))*gen.sigma0
            dE = E[b]/gen.sqrt_psd_g + (rng.standard_normal(gen.n_bins)+1j*rng.standard_normal(gen.n_bins))*gen.sigma0

            nrng = np.random.RandomState(500 + SB + done)
            ens = []
            ref0 = None
            base_ref = draw_ref(P[b], t_pre[b], kbase, nrng, s=args.ref_scale,
                                wide_spin=args.wide_spin_ref)  # single Mc/q/t reference
            if args.refine:
                # round-0 with the drawn ref -> intrinsic medians -> refined ref
                feats = het_features(gen, dA, dE, base_ref)
                fd = torch.as_tensor(cp.asnumpy(feats), dtype=torch.float32,
                                     device=device).unsqueeze(0)
                if args.n_ucb > 0:
                    g = torch.Generator(device=device).manual_seed(done * 7 + 999)
                    inject_ucb_foreground(fd, torch.tensor([args.n_ucb], device=device),
                                          g_bin, band, rng=g)
                fd = (fd - fmean) / fstd
                base = torch.from_numpy(ref_meta_vec(base_ref)).unsqueeze(0).to(device).float()
                if cfg.get('ucb_cond', False):
                    base = torch.cat([base, torch.tensor([[args.n_ucb / UCB_N_MAX_COND]],
                                                         device=device)], dim=-1)
                torch.manual_seed((SB + done) * 31 + 777)
                samp = gnpe_marg(m, fd, base, device, args.rounds, args.n_per,
                                 args.inflate, seed=(SB + done) * 31 + 777, k_pose=K_POSE)
                ph0 = residual_to_physical(samp, base_ref, kbase, params=res_p)
                T_OBS_HR = 2.6298e6 / 3600.0
                base_ref = dict(Mc=float(np.median(ph0['Mc'])),
                                q=float(max(np.median(ph0['q']), 1.0001)),
                                c1=float(np.clip(np.median(ph0['c1']), -0.99, 0.99)),
                                c2=float(np.clip(np.median(ph0['c2']), -0.99, 0.99)),
                                t_ref_hr=T_OBS_HR - float(np.median(ph0['t_pre_hours'])),
                                kernel_scale=args.refine_scale,
                                s_phase=args.refine_scale, kernel_type=0)
            for kr in range(K_REF):
                if args.refine:
                    ref = base_ref                  # the self-refined reference
                elif args.spin_marg:
                    # share Mc/q/t; marginalize ONLY the (uninformative) spin ref
                    ref = dict(base_ref)
                    ref['c1'] = float(nrng.uniform(-0.99, 0.99))
                    ref['c2'] = float(nrng.uniform(-0.99, 0.99))
                else:
                    ref = draw_ref(P[b], t_pre[b], kbase, nrng, s=args.ref_scale,
                                   wide_spin=args.wide_spin_ref)
                if kr == 0:
                    ref0 = ref
                feats = het_features(gen, dA, dE, ref)
                fd = torch.as_tensor(cp.asnumpy(feats), dtype=torch.float32,
                                     device=device).unsqueeze(0)
                if args.n_ucb > 0:
                    g = torch.Generator(device=device).manual_seed(done * 7 + kr)
                    inject_ucb_foreground(fd, torch.tensor([args.n_ucb], device=device),
                                          g_bin, band, rng=g)
                fd = (fd - fmean) / fstd
                base = torch.from_numpy(ref_meta_vec(ref)).unsqueeze(0).to(device).float()
                if cfg.get('ucb_cond', False):
                    base = torch.cat([base, torch.tensor([[args.n_ucb / UCB_N_MAX_COND]],
                                                         device=device)], dim=-1)
                torch.manual_seed((SB + done) * 31 + kr)     # deterministic flow draws
                samp = gnpe_marg(m, fd, base, device, args.rounds, args.n_per,
                                 args.inflate, seed=(SB + done) * 31 + kr, k_pose=K_POSE)
                # -> PHYSICAL 11D (ref-specific residual mapping), PARAM_NAMES order
                phys = residual_to_physical(samp, ref, kbase, params=res_p)
                lo, hi = PARAM_RANGES['logD']
                mem = np.column_stack([
                    phys['Mc'], phys['q'], phys['c1'], phys['c2'],
                    samp[:, 4]*(hi-lo)+lo, samp[:, 5]*2-1, samp[:, 6]*2-1,
                    samp[:, 7]*2*np.pi, samp[:, 8]*np.pi, samp[:, 9]*2*np.pi,
                    phys['t_pre_hours']])
                ens.append(mem)
            # ONE percentile of physical truth in the COMBINED marginalized posterior
            phys_all = np.concatenate(ens, axis=0)
            truth_phys = np.array([P[b, 0], P[b, 1], P[b, 2], P[b, 3], P[b, 4],
                                   P[b, 6], P[b, 7], P[b, 8], P[b, 9], P[b, 10], t_pre[b]])
            for j, n in enumerate(PARAM_NAMES):
                pct[n].append((phys_all[:, j] <= truth_phys[j]).mean())
            # chi_eff = (q*c1 + c2)/(q+1) — the measurable spin combination (q=m1/m2)
            qcol = phys_all[:, 1]
            chieff = (qcol * phys_all[:, 2] + phys_all[:, 3]) / (qcol + 1.0)
            chieff_t = (P[b, 1] * P[b, 2] + P[b, 3]) / (P[b, 1] + 1.0)
            pct.setdefault('chi_eff', []).append((chieff <= chieff_t).mean())
            # joint 2D sky: HPD credibility level at which the true position enters
            # the credible region (smoothed histogram estimator, lambda wrapped)
            from scipy.ndimage import gaussian_filter as _gf
            NL, NS, SM_SKY = 90, 45, 1.5
            Hs2, _, _ = np.histogram2d(phys_all[:, 7] % (2*np.pi), phys_all[:, 6],
                                       bins=[NL, NS], range=[[0, 2*np.pi], [-1, 1]])
            pad = 8
            Hp = np.vstack([Hs2[-pad:], Hs2, Hs2[:pad]])        # wrap lambda axis
            Hp = _gf(Hp, SM_SKY, mode='nearest')[pad:-pad]
            Hp = Hp / Hp.sum()
            it = min(int((truth_phys[7] % (2*np.pi)) / (2*np.pi) * NL), NL - 1)
            jt = min(int((truth_phys[6] + 1.0) / 2.0 * NS), NS - 1)
            sky_cred.append(float(Hp[Hp > Hp[it, jt]].sum()))
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{args.n_events}", flush=True)

    N = done
    # save raw percentile ranks for the calibration-summary figures
    np.savez(args.out_ranks, N=N, n_ucb=args.n_ucb, sky_cred=np.array(sky_cred),
             **{n: np.array(pct[n]) for n in pct})
    sc = np.array(sky_cred)
    print(f"saved {args.out_ranks} ({N} events)")
    print(f"2D sky coverage: 50% region -> {100*(sc <= 0.5).mean():.1f}%   "
          f"90% region -> {100*(sc <= 0.9).mean():.1f}%", flush=True)
    from scipy.stats import kstest
    _paper_style()
    LBL = {'Mc': r'$\mathcal{M}_c$', 'q': r'$q$', 'c1': r'$\chi_1$', 'c2': r'$\chi_2$',
           'chi_eff': r'$\chi_{\rm eff}$', 'logD': r'$\log_{10}D_L$', 'cos_inc': r'$\cos\iota$',
           'sin_beta': r'$\sin\beta$', 'lambda': r'$\lambda$', 'psi': r'$\psi$',
           'phi0': r'$\phi_0$', 't_pre_hours': r'$t_c$'}
    PLOT_NAMES = ['Mc', 'q', 'c1', 'c2', 'logD', 'cos_inc', 'sin_beta',
                  'lambda', 'psi', 'phi0', 't_pre_hours']
    PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
               '#e377c2', '#17becf', '#bcbd22', '#7f7f7f', '#393b79']
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    xp = np.linspace(0, 1, 200); sci = np.sqrt(xp * (1 - xp) / N)
    for k, a in zip((1, 2, 3), (0.18, 0.12, 0.07)):
        ax.fill_between(xp, np.clip(xp - k*sci, 0, 1), np.clip(xp + k*sci, 0, 1),
                        color='0.5', alpha=a, lw=0)
    y = np.arange(1, N + 1) / N
    for n, c in zip(PLOT_NAMES, PALETTE):
        p = np.sort(np.array(pct[n])); pv = kstest(pct[n], 'uniform').pvalue
        ax.plot(p, y, color=c, lw=1.6, label=f"{LBL[n]}  ($p={pv:.2f}$)", solid_capstyle='round')
    ax.plot([0, 1], [0, 1], color='k', ls=(0, (4, 3)), lw=1.1, alpha=0.7, zorder=1)
    ax.set_xlabel('Theoretical CDF'); ax.set_ylabel('Empirical CDF')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_title(r'\textbf{HERON} calibration under galactic foreground'
                 if plt.rcParams['text.usetex'] else 'HERON calibration under galactic foreground',
                 pad=10)
    ax.legend(fontsize=8.5, ncol=2, loc='upper left', frameon=False,
              handlelength=1.4, columnspacing=1.0, labelspacing=0.3)
    ax.text(0.97, 0.03, f'$N={N}$ injections, $n_{{\\rm UCB}}={args.n_ucb}$',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='0.35')
    fig.tight_layout()
    fig.savefig(figure_out('results_pp_marginalized.png'), dpi=300, bbox_inches='tight')
    fig.savefig(figure_out('fig_pp.pdf'), bbox_inches='tight')
    print("KS p-values:", "  ".join(
        f"{n}={kstest(np.array(pct[n]),'uniform').pvalue:.2f}"
        for n in PLOT_NAMES + ['chi_eff']))
    print(f"saved {figure_out('fig_pp.pdf')} (raw P-P; the published Fig. 3a is the recalibrated one, evaluation/fit_recalibration.py)")


if __name__ == '__main__':
    main()
