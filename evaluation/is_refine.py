"""Idea B — IMPORTANCE SAMPLING on the pooled reference-marginalized posterior.

The K_ref-pooled posterior is over-dispersed on the intrinsics (P-P rank-std
0.09-0.25 vs 0.289 uniform) — which makes it an IDEAL importance-sampling
proposal: it safely covers the exact posterior. Reweighting with the exact
whitened-residual likelihood then gives the exact posterior by construction
(both corner AND coverage), a la DINGO-IS.

Since GNPE samples carry no tractable density, the proposal density is a GMM
fit to the pooled samples; fresh samples are drawn FROM the GMM (so q is exact
for the points we weight). Prior taken uniform in the fitted coordinates.

Reports: ESS (the go/no-go number), IS-corrected 68% CIs vs truth.
Run: python evaluation/is_refine.py --k-ref 16
"""
import sys, json, argparse
import numpy as np, torch, cupy as cp
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from heron.artifacts import MBHB_FLOW, MBHB_NORM, load_kernel, product_out
from pe_smbhb.data_mbhb import HetDataGenerator
from pe_smbhb.train_flow import build_model, ref_meta_vec, residual_to_physical
from pe_smbhb.networks import PARAM_RANGES
from pe_smbhb.inference import draw_ref, het_features, gnpe_marg
from heron.sources import SANGRIA as CASES

T_OBS_HR = 2.6298e6 / 3600.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k-ref', type=int, default=16)
    ap.add_argument('--n-per', type=int, default=1000)
    ap.add_argument('--n-is', type=int, default=4096, help='IS proposal draws')
    ap.add_argument('--gmm-comps', type=int, default=32)
    ap.add_argument('--model', default=MBHB_FLOW)
    ap.add_argument('--norm', default=MBHB_NORM)
    ap.add_argument('--data', default=None)
    ap.add_argument('--t-pre', type=float, default=24.0)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
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

    S = CASES[1]['src']
    m1, m2 = S['m1'], S['m2']; eta = m1*m2/(m1+m2)**2
    Mc, q = (m1+m2)*eta**0.6, m1/m2
    t_ref = (gen.t_obs - args.t_pre*3600)/3600
    trow = np.array([Mc, q, S['chi1'], S['chi2'], np.log10(S['dist_Mpc']), t_ref,
                     np.cos(S['iota']), np.sin(S['beta']), S['lam'], S['psi'], S['phi0']])
    A, E = gen.gen_wf(trow[None])
    rng = cp.random.RandomState(0)
    dA = A[0]/gen.sqrt_psd_g + (rng.standard_normal(gen.n_bins)+1j*rng.standard_normal(gen.n_bins))*gen.sigma0
    dE = E[0]/gen.sqrt_psd_g + (rng.standard_normal(gen.n_bins)+1j*rng.standard_normal(gen.n_bins))*gen.sigma0
    sig2 = float(gen.sigma0)**2
    lo, hi = PARAM_RANGES['logD']

    # pooled K_ref proposal samples (physical 11D)
    nrng = np.random.RandomState(7)
    parts = []
    for kr in range(args.k_ref):
        ref = draw_ref(trow, args.t_pre, kbase, nrng, s=1.0, wide_spin=True)
        feats = het_features(gen, dA, dE, ref)
        fd = torch.as_tensor(cp.asnumpy(feats), dtype=torch.float32, device=device).unsqueeze(0)
        fd = (fd - fmean) / fstd
        base = torch.from_numpy(ref_meta_vec(ref)).unsqueeze(0).to(device).float()
        if cfg.get('ucb_cond', False):
            base = torch.cat([base, torch.tensor([[0.0]], device=device)], dim=-1)
        samp = gnpe_marg(m, fd, base, device, 3, args.n_per, 1.6, seed=13+kr)
        ph = residual_to_physical(samp, ref, kbase, params=res_p)
        parts.append(np.column_stack([
            ph['Mc'], ph['q'], ph['c1'], ph['c2'], samp[:, 4]*(hi-lo)+lo,
            T_OBS_HR - ph['t_pre_hours'], samp[:, 5]*2-1, samp[:, 6]*2-1,
            samp[:, 7]*2*np.pi, samp[:, 8]*np.pi, samp[:, 9]*2*np.pi]))
    pool = np.concatenate(parts, axis=0)   # cols: Mc q c1 c2 logD t_ref ci sb lam psi phi0
    print(f"pooled proposal: {pool.shape[0]} samples from K_ref={args.k_ref}")

    # GMM proposal density in standardized coords
    from sklearn.mixture import GaussianMixture
    X = pool.copy(); X[:, 0] = np.log10(X[:, 0])          # log10 Mc
    mu, sd = X.mean(0), X.std(0)
    gmm = GaussianMixture(n_components=args.gmm_comps, covariance_type='full',
                          reg_covar=1e-6, random_state=0, max_iter=300)
    gmm.fit((X - mu) / sd)
    Z, _ = gmm.sample(args.n_is * 2)
    Y = Z * sd + mu
    ok = ((Y[:, 1] >= 1.0001) & (np.abs(Y[:, 2]) <= 0.99) & (np.abs(Y[:, 3]) <= 0.99)
          & (np.abs(Y[:, 6]) <= 1.0) & (np.abs(Y[:, 7]) <= 1.0)
          & (Y[:, 8] >= 0) & (Y[:, 8] <= 2*np.pi)
          & (Y[:, 9] >= 0) & (Y[:, 9] <= np.pi)
          & (Y[:, 10] >= 0) & (Y[:, 10] <= 2*np.pi))
    Y = Y[ok][:args.n_is]; Z = Z[ok][:args.n_is]
    logq = gmm.score_samples(Z)
    print(f"IS draws kept: {len(Y)} (accept {ok.mean():.2f})")

    # exact whitened-residual lnL, batched via bbhx
    P = np.empty((len(Y), 11))
    P[:, 0] = 10**Y[:, 0]; P[:, 1:5] = Y[:, 1:5]; P[:, 5] = Y[:, 5]
    P[:, 6:] = Y[:, 6:]
    lnL = np.empty(len(Y))
    B = 64
    for i in range(0, len(Y), B):
        Ah, Eh = gen.gen_wf(P[i:i+B])
        rA = dA[None, :] - Ah/gen.sqrt_psd_g[None, :]
        rE = dE[None, :] - Eh/gen.sqrt_psd_g[None, :]
        lnL[i:i+B] = cp.asnumpy(-0.5*(cp.sum(cp.abs(rA)**2, axis=1)
                                      + cp.sum(cp.abs(rE)**2, axis=1))/sig2)
    lnw = lnL - logq          # prior ~ uniform in fitted coords
    lnw -= lnw.max()
    w = np.exp(lnw); w /= w.sum()
    ess = 1.0/np.sum(w**2)
    print(f"\nESS = {ess:.1f} / {len(w)}  ({100*ess/len(w):.1f}%)")

    # IS-corrected CIs
    def wq(x, ws, qs):
        i = np.argsort(x); c = np.cumsum(ws[i])
        return np.interp(qs, c/c[-1], x[i])
    ce = (P[:, 1]*P[:, 2]+P[:, 3])/(P[:, 1]+1.0)
    ceT = (q*S['chi1']+S['chi2'])/(q+1)
    print(f"{'param':8s} {'truth':>9s} {'proposal 68%':>22s} {'IS-corrected 68%':>22s}")
    for nm_, col, tr in [('Mc', P[:, 0]/1e6, Mc/1e6), ('q', P[:, 1], q),
                         ('chi_eff', ce, ceT), ('logD', P[:, 4], np.log10(S['dist_Mpc']))]:
        p = np.percentile(col, [16, 50, 84])
        c16, c50, c84 = wq(col, w, [0.16, 0.5, 0.84])
        print(f"{nm_:8s} {tr:9.4f}  [{p[0]:8.4f},{p[2]:8.4f}]  [{c16:8.4f},{c84:8.4f}]"
              f"  med {c50:8.4f}")
    out = product_out('is_refine_case1.npz')
    np.savez(out, P=P, w=w, lnL=lnL, logq=logq, ess=ess)
    print(f"saved {out}")


if __name__ == '__main__':
    main()
