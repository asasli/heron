"""
The two-tier corner (Fig. 2): the Sangria Case-1 event through both latency
tiers of the pipeline, on the sky-frame-independent parameters
[Mc, q, chi_eff, log D_L, cos(iota)].

  red   HERON alert: amortized posterior, one reference, post-hoc calibrated
  blue  stage 4: alert-seeded exact inference on the stage-3 subtracted data
        (paper_products/pointing/synergy_subtracted_alert.npz)
  grey  unseeded chain at matched wall time (synergy_subtracted_none.npz)

The alert posterior can be taken from two places:

  --alert recompute (default; the published figure)
      The alert is recomputed here for the same binary and the same instrument
      noise realization (seed 0), with the foreground represented by `--n-ucb`
      resolved lines injected through the calibrated per-bin response
      (weights/ucb_response.npz), as in the calibration runs.
  --alert handoff
      The alert the pipeline itself produced on the foreground data set that
      stage 3 subtracts (30 UCB waveforms + confusion realization): the
      `alert_samples` stored in the hand-off file, i.e. the draws that seeded
      the blue chain.

Run: python evaluation/fig_two_tier_final.py --n-ucb 30          (needs a GPU)
     python evaluation/fig_two_tier_final.py --alert handoff     (no GPU)
"""
import sys, json, argparse
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from heron.artifacts import (MBHB_FLOW, MBHB_NORM, MBHB_RECAL, UCB_RESPONSE, load_kernel,
                             product_in, figure_out)
from heron.sources import SANGRIA as CASES
from heron.recal import load_recal, recalibrate_samples

RECAL_COLS = ['Mc', 'q', 'c1', 'c2', 'logD', 'cos_inc', 'sin_beta',
              'lambda', 'psi', 'phi0']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alert', choices=['recompute', 'handoff'], default='recompute')
    ap.add_argument('--handoff', default=None,
                    help='hand-off npz with alert_samples (default: the published one)')
    ap.add_argument('--n-ucb', type=int, default=30)
    ap.add_argument('--stage4', default=None)
    ap.add_argument('--stage4-label', default='HyperWave (PTMCMC + learned proposals),\nHERON-seeded, 19 min')
    ap.add_argument('--unseeded', default=None,
                    help="unseeded chain at matched wall-time ('' to omit)")
    ap.add_argument('--unseeded-label', default='Eryn PTMCMC (current public sampler),\nunseeded, 15 min')
    ap.add_argument('--model', default=MBHB_FLOW)
    ap.add_argument('--norm', default=MBHB_NORM)
    ap.add_argument('--data', default=None)
    ap.add_argument('--recal-map', default=MBHB_RECAL)
    ap.add_argument('--t-pre', type=float, default=24.0)
    ap.add_argument('--n-per', type=int, default=4000)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    args.stage4 = args.stage4 or product_in('pointing/synergy_subtracted_alert.npz')
    if args.unseeded is None:
        args.unseeded = product_in('pointing/synergy_subtracted_none.npz')
    recal = load_recal(args.recal_map)

    # stage-4 chain (frame-independent columns)
    z4 = np.load(args.stage4)
    s4 = z4['samples'].astype(float)   # Mc q chi1z chi2z log10_D tc_h cosi sinb lam psi phi
    ce4 = (s4[:, 1]*s4[:, 2] + s4[:, 3])/(s4[:, 1] + 1.0)
    blue = np.column_stack([s4[:, 0]/1e6, s4[:, 1], ce4, s4[:, 4], s4[:, 6]])
    wall4 = float(z4['wall'])/60

    S = CASES[1]['src']
    m1, m2 = S['m1'], S['m2']; eta = m1*m2/(m1+m2)**2
    Mc, q = (m1+m2)*eta**0.6, m1/m2
    if args.alert == 'handoff':
        hf = args.handoff or product_in('handoff/sangria_synergy_subtracted.npz')
        a = np.load(hf, allow_pickle=True)['alert_samples'].astype(float)
        ours10 = a[:, :10]      # Mc q chi1 chi2 logD cos_iota sin_beta lambda psi phi0
        args.n_ucb = 30
    else:
        ours10 = recompute_alert(args, S, Mc, q)
    ce = (ours10[:, 1]*ours10[:, 2]+ours10[:, 3])/(ours10[:, 1]+1.0)
    if recal is not None:
        ce = recalibrate_samples(ce[:, None], {'x': recal['chi_eff']}, ['x'])[:, 0]
        ours10 = recalibrate_samples(ours10, recal, RECAL_COLS)
        print("[recal] applied")
    red = np.column_stack([ours10[:, 0]/1e6, ours10[:, 1], ce, ours10[:, 4], ours10[:, 5]])
    plot(args, red, blue, wall4, S, Mc, q)


def recompute_alert(args, S, Mc, q):
    """The published alert posterior of Fig. 2 (see the module docstring)."""
    import torch, cupy as cp
    from pe_smbhb.data_mbhb import HetDataGenerator
    from pe_smbhb.train_flow import build_model, ref_meta_vec, residual_to_physical
    from pe_smbhb.networks import PARAM_RANGES
    from pe_smbhb.inference import draw_ref, het_features, gnpe_marg
    torch.manual_seed(0)
    device = torch.device(args.device)
    ck = torch.load(args.model, map_location=device, weights_only=False)
    cfg = ck['config']; m = build_model(cfg).to(device)
    m.load_state_dict(ck['model_state_dict']); m.eval()
    res_p = cfg.get('residual_params') or None
    nm = torch.load(args.norm, map_location=device, weights_only=False)
    fmean, fstd = nm['fd_mean'].to(device), nm['fd_std'].to(device)
    kbase = load_kernel(args.data)

    gen = HetDataGenerator(output_dir=None)
    np.random.seed(123)
    for _ in range(2):
        Pw, _ = gen.draw_priors(64); gen.gen_wf(Pw)

    t_ref = (gen.t_obs - args.t_pre*3600)/3600
    trow = np.array([Mc, q, S['chi1'], S['chi2'], np.log10(S['dist_Mpc']), t_ref,
                     np.cos(S['iota']), np.sin(S['beta']), S['lam'], S['psi'], S['phi0']])
    A, E = gen.gen_wf(trow[None])
    rng = cp.random.RandomState(0)          # seed-0 noise = the export realization
    dA = A[0]/gen.sqrt_psd_g + (rng.standard_normal(gen.n_bins)+1j*rng.standard_normal(gen.n_bins))*gen.sigma0
    dE = E[0]/gen.sqrt_psd_g + (rng.standard_normal(gen.n_bins)+1j*rng.standard_normal(gen.n_bins))*gen.sigma0

    g_bin = band = None
    if args.n_ucb > 0:
        from pe_smbhb.train_flow import inject_ucb_foreground, UCB_N_MAX_COND
        resp = np.load(UCB_RESPONSE)
        g_bin = torch.from_numpy(resp['g_bin'].astype(np.float32)).to(device)
        band = torch.from_numpy(np.where(resp['g_bin'] > 0)[0].astype(np.int64)).to(device)

    nrng = np.random.RandomState(7)
    ref = draw_ref(trow, args.t_pre, kbase, nrng, s=0.45, wide_spin=True)
    feats = het_features(gen, dA, dE, ref)
    fd = torch.as_tensor(cp.asnumpy(feats), dtype=torch.float32, device=device).unsqueeze(0)
    if args.n_ucb > 0:
        g = torch.Generator(device=device).manual_seed(3)
        inject_ucb_foreground(fd, torch.tensor([args.n_ucb], device=device), g_bin, band, rng=g)
    fd = (fd - fmean) / fstd
    base = torch.from_numpy(ref_meta_vec(ref)).unsqueeze(0).to(device).float()
    if cfg.get('ucb_cond', False):
        from pe_smbhb.train_flow import UCB_N_MAX_COND
        base = torch.cat([base, torch.tensor([[args.n_ucb / UCB_N_MAX_COND]],
                                             device=device)], dim=-1)
    samp = gnpe_marg(m, fd, base, device, 3, args.n_per, 1.6, seed=13)
    phys = residual_to_physical(samp, ref, kbase, params=res_p)
    lo, hi = PARAM_RANGES['logD']
    ours10 = np.column_stack([
        phys['Mc'], phys['q'], phys['c1'], phys['c2'], samp[:, 4]*(hi-lo)+lo,
        samp[:, 5]*2-1, samp[:, 6]*2-1, samp[:, 7]*2*np.pi, samp[:, 8]*np.pi,
        samp[:, 9]*2*np.pi])
    return ours10


def plot(args, red, blue, wall4, S, Mc, q):
    # corner
    LAB = [r'$\mathcal{M}_c/10^6$', r'$q$', r'$\chi_{\rm eff}$',
           r'$\log D_L$', r'$\cos\iota$']
    ceT = (q*S['chi1']+S['chi2'])/(q+1)
    TR = [Mc/1e6, q, ceT, np.log10(S['dist_Mpc']), np.cos(S['iota'])]
    hero = r'HERON alert ($<1$ s)' + (f', $n_{{\\rm UCB}}={args.n_ucb}$' if args.n_ucb > 0 else '')
    title = None  # no suptitle: bare corner with legend only

    grey = None
    if args.unseeded:
        zu = np.load(args.unseeded)
        su = zu['samples'].astype(float)
        ceu = (su[:, 1]*su[:, 2] + su[:, 3])/(su[:, 1] + 1.0)
        grey = np.column_stack([su[:, 0]/1e6, su[:, 1], ceu, su[:, 4], su[:, 6]])

    # extents locked to alert+seeded+truth (the unseeded chain spills off-plot
    # by design: at matched wall-time it has not converged)
    ext = {}
    for i, lab in enumerate(LAB):
        lo = min(np.percentile(red[:, i], 0.2), np.percentile(blue[:, i], 0.2), TR[i])
        hi = max(np.percentile(red[:, i], 99.8), np.percentile(blue[:, i], 99.8), TR[i])
        pad = 0.08*(hi - lo)
        ext[lab] = (lo - pad, hi + pad)

    import matplotlib.pyplot as plt
    from chainconsumer import ChainConsumer
    from evaluation.plotstyle import CORNFLOWER, HERON_CORAL, paper_style
    paper_style()
    c = ChainConsumer()
    if grey is not None:
        c.add_chain(grey, parameters=LAB, name=args.unseeded_label)
    c.add_chain(blue, parameters=LAB, name=args.stage4_label)
    c.add_chain(red, parameters=LAB, name=hero)
    ncol = 3 if grey is not None else 2
    colors = (['#9a9a9a'] if grey is not None else []) + [CORNFLOWER, HERON_CORAL]
    c.configure(colors=colors, linewidths=[1.1]*ncol,
                shade=([False] if grey is not None else []) + [True, True],
                shade_alpha=([0.0] if grey is not None else []) + [0.55, 0.30],
                sigmas=[1, 2], kde=False, smooth=1, bar_shade=False, serif=True,
                statistics='mean', legend_kwargs={'fontsize': 12, 'loc': 'upper right'},
                label_font_size=15, tick_font_size=10, max_ticks=4, summary=False,
                diagonal_tick_labels=False, spacing=0.0)
    fig = c.plotter.plot(figsize=(7.5, 7.5), extents=ext,
                         truth={p: t for p, t in zip(LAB, TR)})
    tag = '' if args.alert == 'recompute' else '_handoff_alert'
    png = figure_out(f'fig_corner_case1_final_twotier{tag}.png')
    fig.savefig(png, dpi=200, bbox_inches='tight')
    pdf = figure_out(f'fig_corner_case1_final_twotier{tag}.pdf')
    fig.savefig(pdf, bbox_inches='tight')
    print(f"saved {png} + {pdf}   (stage-4 wall {wall4:.1f} min)")
    print(f"{'param':8s} {'truth':>9s} {'alert 68%':>22s} {'HyperWave 68%':>22s}")
    for i, nm_ in enumerate(['Mc', 'q', 'chi_eff', 'logD', 'cosi']):
        rl, rh = np.percentile(red[:, i], [16, 84])
        bl, bh = np.percentile(blue[:, i], [16, 84])
        print(f"{nm_:8s} {TR[i]:9.3f}  [{rl:8.3f},{rh:8.3f}]  [{bl:8.3f},{bh:8.3f}]")


if __name__ == '__main__':
    main()
