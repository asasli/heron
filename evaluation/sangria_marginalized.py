"""
Sangria corner + MCMC overlay using the GNPE-MARGINALIZED inference
(K_REF reference draws x K_POSE pose draws), so the posterior is properly
dispersed (fixes the single-draw artifact: reference-anchored spins that looked
artificially tight vs MCMC). Produces overlay_sangria_marginalized.png.

Run: python evaluation/sangria_marginalized.py \
        --model <ckpt> --norm <norm> --ref-scale 0.45
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
import h5py, corner as corner_mod
from pe_smbhb.data_mbhb import HetDataGenerator
from pe_smbhb.train_flow import (build_model, ref_meta_vec, residual_to_physical, sky_area_90)
from pe_smbhb.networks import PARAM_RANGES
from evaluation.probe_oracle_ref import build_fd
from heron.sources import SANGRIA
from pe_smbhb.inference import draw_ref, het_features, gnpe_marg

# Sangria sources (heron/sources.py) and their exact reference posteriors:
# thinned cold-chain samples of the parallel-tempered MCMC runs of
# arXiv:2602.22074, stored in paper_products/reference_chains/.
CASES = {k: dict(src=v['src'], name=v['name'], mcmc=f'reference_chains/case{k}_ptmcmc.npz')
         for k, v in SANGRIA.items()}
LAB = [r'$\mathcal{M}_c/10^6$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\log D_L$',
       r'$\cos\iota$', r'$\sin\beta$', r'$\lambda$', r'$\psi$', r'$\phi_0$']
MCMC_COLS = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]


def load_mcmc(MCMC, n=60000):
    """Reference posterior samples, columns [Mc/1e6, q, chi1, chi2, log10 D_L,
    cos(iota), sin(beta), lambda, psi, phi0]."""
    return np.load(product_in(MCMC))['samples'][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=MBHB_FLOW)
    ap.add_argument('--norm', default=MBHB_NORM)
    ap.add_argument('--data', default=None)
    ap.add_argument('--t-pre', type=float, default=24.0)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--inflate', type=float, default=1.6)
    ap.add_argument('--ref-scale', type=float, default=0.45)
    ap.add_argument('--k-ref', type=int, default=8)
    ap.add_argument('--n-per', type=int, default=4000)
    ap.add_argument('--case', type=int, default=1, choices=[1, 2, 3])
    ap.add_argument('--n-ucb', type=int, default=0, help='inject galactic foreground (real scenario)')
    ap.add_argument('--clean', action='store_true', help='main-text subset corner: Mc,q,chi_eff,logD,sinb,lam')
    ap.add_argument('--wide-spin-ref', action='store_true', help='uninformative reference spins (for WIDE_SPIN_REF-trained v15+)')
    ap.add_argument('--recal-map', default=MBHB_RECAL, help='post-hoc calibration map')
    ap.add_argument('--two-tier', action='store_true',
                    help='label the overlay as the pipeline tiers: HERON alert (<1 s) vs HyperWave (exact)')
    ap.add_argument('--ref-label', default=None,
                    help="reference-chain legend label, e.g. 'HyperWave (HERON-seeded, ~20 min)'")
    ap.add_argument('--out-tag', default='', help='suffix for the output filename')
    ap.add_argument('--no-recal', action='store_true', help='disable post-hoc calibration')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    device = torch.device(args.device)
    SANGRIA = CASES[args.case]['src']; MCMC = CASES[args.case]['mcmc']
    CNAME = CASES[args.case]['name']

    ck = torch.load(args.model, map_location=device, weights_only=False)
    cfg = ck['config']
    m = build_model(cfg).to(device); m.load_state_dict(ck['model_state_dict']); m.eval()
    res_p = cfg.get('residual_params') or None  # v14abs: only Mc,t residual; q/chi absolute
    nm = torch.load(args.norm, map_location=device, weights_only=False)
    fmean, fstd = nm['fd_mean'].to(device), nm['fd_std'].to(device)
    kbase = load_kernel(args.data)

    gen = HetDataGenerator(output_dir=None)
    np.random.seed(123)
    for _ in range(2):
        Pw, _ = gen.draw_priors(64); gen.gen_wf(Pw)

    m1, m2 = SANGRIA['m1'], SANGRIA['m2']; eta = m1*m2/(m1+m2)**2
    Mc = (m1+m2)*eta**0.6; q = m1/m2
    t_ref_hr = (gen.t_obs - args.t_pre*3600)/3600
    p_true = np.array([[Mc, q, SANGRIA['chi1'], SANGRIA['chi2'], np.log10(SANGRIA['dist_Mpc']),
                        t_ref_hr, np.cos(SANGRIA['iota']), np.sin(SANGRIA['beta']),
                        SANGRIA['lam'], SANGRIA['psi'], SANGRIA['phi0']]])
    A, E = gen.gen_wf(p_true)
    rng = cp.random.RandomState(0)
    dA = A[0]/gen.sqrt_psd_g + (rng.standard_normal(gen.n_bins)+1j*rng.standard_normal(gen.n_bins))*gen.sigma0
    dE = E[0]/gen.sqrt_psd_g + (rng.standard_normal(gen.n_bins)+1j*rng.standard_normal(gen.n_bins))*gen.sigma0
    snr = float(cp.sqrt(4*gen.df*cp.sum((cp.abs(A[0])**2+cp.abs(E[0])**2)/gen.psd_g)))
    print(f"{CNAME} SNR={snr:.0f}; marginalized K_REF={args.k_ref} ref-scale={args.ref_scale}")

    # optional galactic UCB foreground (REAL scenario: data contains confusion)
    g_bin = band = None
    if args.n_ucb > 0:
        from pe_smbhb.train_flow import inject_ucb_foreground, UCB_N_MAX_COND
        resp = np.load(UCB_RESPONSE)
        g_bin = torch.from_numpy(resp['g_bin'].astype(np.float32)).to(device)
        band = torch.from_numpy(np.where(resp['g_bin'] > 0)[0].astype(np.int64)).to(device)
        print(f"  injecting galactic foreground: n_ucb={args.n_ucb} resolved + faint confusion")

    # marginalized ensemble
    trow = p_true[0]
    nrng = np.random.RandomState(7)
    parts = []
    for kr in range(args.k_ref):
        ref = draw_ref(trow, args.t_pre, kbase, nrng, s=args.ref_scale, wide_spin=args.wide_spin_ref)
        feats = het_features(gen, dA, dE, ref)
        fd = torch.as_tensor(cp.asnumpy(feats), dtype=torch.float32, device=device).unsqueeze(0)
        if args.n_ucb > 0:
            g = torch.Generator(device=device).manual_seed(kr * 17 + 3)
            inject_ucb_foreground(fd, torch.tensor([args.n_ucb], device=device), g_bin, band, rng=g)
        fd = (fd - fmean) / fstd
        base = torch.from_numpy(ref_meta_vec(ref)).unsqueeze(0).to(device).float()
        if cfg.get('ucb_cond', False):
            from pe_smbhb.train_flow import UCB_N_MAX_COND
            base = torch.cat([base, torch.tensor([[args.n_ucb / UCB_N_MAX_COND]],
                                                 device=device)], dim=-1)
        samp = gnpe_marg(m, fd, base, device, args.rounds, args.n_per, args.inflate, seed=kr*13)
        phys = residual_to_physical(samp, ref, kbase, params=res_p)
        lo, hi = PARAM_RANGES['logD']
        parts.append(np.column_stack([
            phys['Mc']/1e6, phys['q'], phys['c1'], phys['c2'], samp[:, 4]*(hi-lo)+lo,
            samp[:, 5]*2-1, samp[:, 6]*2-1, samp[:, 7]*2*np.pi, samp[:, 8]*np.pi,
            samp[:, 9]*2*np.pi]))
    ours = np.concatenate(parts, axis=0)
    # post-hoc calibration (same map applied automatically at every alert);
    # rank-based -> scale-invariant, order-preserving -> correlations intact
    from pe_smbhb.train_flow import load_recal, recalibrate_samples
    recal = None if args.no_recal else load_recal(args.recal_map)
    chi_recal = None
    if recal is not None:
        col_order = ['Mc', 'q', 'c1', 'c2', 'logD', 'cos_inc', 'sin_beta',
                     'lambda', 'psi', 'phi0']
        # recalibrate chi_eff on its own map BEFORE remapping c1/c2/q (derived)
        if 'chi_eff' in recal:
            ce = ((ours[:, 1] * ours[:, 2] + ours[:, 3]) / (ours[:, 1] + 1.0))
            chi_recal = recalibrate_samples(ce[:, None], {'x': recal['chi_eff']}, ['x'])[:, 0]
        ours = recalibrate_samples(ours, recal, col_order)
        print("[recal] applied post-hoc calibration map to the HERON posterior")
    a90 = sky_area_90(ours[:4000, 7], ours[:4000, 6])
    mcmc = load_mcmc(MCMC)
    TRUTH = [Mc/1e6, q, SANGRIA['chi1'], SANGRIA['chi2'], np.log10(SANGRIA['dist_Mpc']),
             np.cos(SANGRIA['iota']), np.sin(SANGRIA['beta']), SANGRIA['lam'],
             SANGRIA['psi'], SANGRIA['phi0']]
    from evaluation.plotstyle import cc_corner
    hero = f'HERON ($n_{{\\rm UCB}}={args.n_ucb}$)' if args.n_ucb > 0 else 'HERON'
    ref_name = 'MCMC (exact)'
    if args.two_tier:
        hero = r'HERON alert ($<1$ s)' + (f' $n_{{\\rm UCB}}={args.n_ucb}$' if args.n_ucb > 0 else '')
        ref_name = r'HyperWave (unseeded, 66 h)'
    if args.ref_label:
        ref_name = args.ref_label
    fgtag = (f"_fg{args.n_ucb}" if args.n_ucb > 0 else "") + args.out_tag

    def chi_eff(a):   # (q*c1 + c2)/(q+1), q = m1/m2 = col1
        return (a[:, 1] * a[:, 2] + a[:, 3]) / (a[:, 1] + 1.0)

    if args.clean:
        # main-text subset: Mc, q, chi_eff, logD, sin_beta, lambda
        LAB_C = [r'$\mathcal{M}_c/10^6$', r'$q$', r'$\chi_{\rm eff}$',
                 r'$\log D_L$', r'$\sin\beta$', r'$\lambda$']
        ce_col = chi_recal if chi_recal is not None else chi_eff(ours)
        o = np.column_stack([ours[:, 0], ours[:, 1], ce_col, ours[:, 4], ours[:, 6], ours[:, 7]])
        mc = np.column_stack([mcmc[:, 0], mcmc[:, 1], chi_eff(mcmc), mcmc[:, 4], mcmc[:, 6], mcmc[:, 7]])
        chieff_t = (q * SANGRIA['chi1'] + SANGRIA['chi2']) / (q + 1.0)
        TR = [Mc/1e6, q, chieff_t, np.log10(SANGRIA['dist_Mpc']),
              np.sin(SANGRIA['beta']), SANGRIA['lam']]
        title = ((f'{CNAME}: alert tier vs pointing tier' if args.two_tier
                  else f'{CNAME}: HERON vs MCMC')
                 + (f'  (foreground $n_{{\\rm UCB}}={args.n_ucb}$)' if args.n_ucb > 0 else ''))
        out = figure_out(f'corner_case{args.case}{fgtag}_clean.png')
        png, pdf = cc_corner(mc, o, LAB_C, TR, out, title=title,
                             mcmc_name=ref_name, ours_name=hero, figsize=7.5)
    else:
        title = f'{CNAME}: HERON vs MCMC   (90% sky ~ {a90:.0f} deg^2)'
        out = figure_out(f'overlay_case{args.case}{fgtag}_marginalized.png')
        png, pdf = cc_corner(mcmc, ours, LAB, TRUTH, out, title=title,
                             mcmc_name=ref_name, ours_name=hero)
    print(f'saved {png} + {pdf}')
    print(f"\n{'param':>6} {'truth':>8} | {'MCMC 68%CI':>20} | {'ours 68%CI':>20}")
    for i, nm_ in enumerate(['Mc','q','c1','c2','logD','cosi','sinb','lam','psi','phi0']):
        ml, mh = np.percentile(mcmc[:, i], [16, 84]); ol, oh = np.percentile(ours[:, i], [16, 84])
        print(f"{nm_:>6} {TRUTH[i]:>8.3g} | [{ml:>8.3g},{mh:>8.3g}] | [{ol:>8.3g},{oh:>8.3g}]")


if __name__ == '__main__':
    main()
