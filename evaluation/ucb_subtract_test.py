"""
UCB SUBTRACTION end-to-end test — the link between the UCB catalog (tier 2)
and the pointing tier (tier 3).

Per fresh source: observe window (signal+noise) -> flow posterior -> subtract
the waveform at the posterior mean -> measure the leftover SIGNAL residual:
  - residual SNR (theory for near-optimal PE: ~sqrt(N_params) ~ 2.6, independent
    of source brightness)
  - local alpha = RMS(residual)/sigma0 over the source's core bins (compare to
    the pointing-tier budget alpha <= 2)
  - global-equivalent alpha for a 40-source foreground over the full band

Run: python evaluation/ucb_subtract_test.py --n 200
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
from pe_ucb.data_ucb import UCBPEGenerator, W, FDOT_LOG10
from pe_ucb.train_flow import UCBFlow, RANGES, PARAMS
from config import *
sys.path.insert(0, NGB_PATH)
from data_utils import ucb


def wf_window_batch(gen, P8, c_bin):
    """MEAN waveform window over a batch of physical param sets P8 (K,8) —
    the MMSE subtraction estimator E[h|d]. Single ucb eval; per-row start_inds
    pasting (the v2 fix). Returns whitened (aw, ew) each (W,) complex."""
    s0 = c_bin - W // 2
    m = ucb(f_lims=np.array([s0 * gen.df, (s0 + W) * gen.df]),
            f_inds=[s0, s0 + W], gpu=True, tobs=gen.t_obs, dt=gen.dt)
    a_r, e_r, _ = m.eval(P8)
    a_r = cp.asarray(a_r); e_r = cp.asarray(e_r)
    si = cp.asnumpy(cp.asarray(m._template.start_inds)).ravel().astype(int)
    K, L = a_r.shape
    a = cp.zeros(W, dtype=cp.complex128); e = cp.zeros(W, dtype=cp.complex128)
    for k in range(K):
        lo, hi = max(s0, si[k]), min(s0 + W, si[k] + L)
        if hi > lo:
            a[lo - s0:hi - s0] += a_r[k, lo - si[k]:hi - si[k]]
            e[lo - s0:hi - s0] += e_r[k, lo - si[k]:hi - si[k]]
    a /= K; e /= K
    w = gen.sqrt_psd_g[s0:s0 + W]
    return a / w, e / w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=UCB_FLOW)
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--polish', action='store_true', help='CEM likelihood polish (flow-seeded)')
    ap.add_argument('--fstat', action='store_true', help='F-statistic subtraction: flow f-seed + analytic 4-amplitude LSQ')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    torch.manual_seed(0)              # deterministic flow draws
    device = torch.device(args.device)

    ck = torch.load(args.model, map_location=device, weights_only=False)
    cfg = ck['config']
    m = UCBFlow(W=cfg['W']).to(device)
    m.load_state_dict(ck['model_state_dict']); m.eval()
    sigma0 = cfg['sigma0']

    gen = UCBPEGenerator(output_dir=None)
    rng = np.random.RandomState(99)
    res_snr, alph, src_snr = [], [], []
    done = 0
    while done < args.n:
        out = gen.one(rng)
        if out is None or out[1]['snr'] < 10:
            continue
        feats, meta = out
        # observe: signal + instrument noise
        g = torch.Generator(device=device).manual_seed(1000 + done)
        x = torch.from_numpy(feats).to(device).unsqueeze(0).clone()
        x += torch.randn(x.shape, generator=g, device=device) * sigma0
        with torch.no_grad():
            s = m.sample(x / sigma0, 64).squeeze(0).cpu().numpy()   # (64, 7)
        s = np.clip(s, 1e-4, 1 - 1e-4)                    # NSF can leak outside [0,1]
        # posterior samples -> physical param batch -> MMSE waveform mean
        phys = np.empty((len(s), 7))
        for j, pnm in enumerate(PARAMS):
            lo, hi = RANGES[pnm]
            phys[:, j] = s[:, j] * (hi - lo) + lo
        P8 = np.column_stack([
            phys[:, 0],                                   # log10A
            (meta['c_bin'] + phys[:, 1]) * gen.df * 1e3,  # f0 [mHz]
            np.full(len(s), FDOT_LOG10),
            phys[:, 2], phys[:, 3], phys[:, 4], phys[:, 5], phys[:, 6]])
        if args.fstat:
            # F-STAT: waveform is LINEAR in 4 amplitude combos of (A,iota,psi,phi0).
            # Numerical basis: 16 random amplitude draws at fixed (f, sky) -> SVD
            # -> rank-4 basis -> closed-form LSQ. Scan f around the flow median;
            # sky from flow samples (weakly coupled in 1 month).
            xw = x.squeeze(0).cpu().numpy()
            d_vec = np.concatenate([xw[:, 0] + 1j*xw[:, 1], xw[:, 2] + 1j*xw[:, 3]])
            med = np.median(s, axis=0)
            lo_, hi_ = RANGES['df_bins']
            df_med = med[1] * (hi_ - lo_) + lo_
            lam_med = med[5] * (RANGES['lambda'][1] - RANGES['lambda'][0]) + RANGES['lambda'][0]
            sb_med = med[6] * 2 - 1
            best_res, best_fit = np.inf, None
            rngf = np.random.default_rng(700 + done)
            df_stage = [df_med + np.linspace(-1.5, 1.5, 13)]
            best_dfb = df_med
            for stage in range(2):
                grid = df_stage[0] if stage == 0 else best_dfb + np.linspace(-0.15, 0.15, 13)
                for dfb in grid:
                    f0g = (meta['c_bin'] + dfb) * gen.df
                    P16 = np.column_stack([
                        np.full(16, -20.8), np.full(16, f0g * 1e3), np.full(16, FDOT_LOG10),
                        rngf.uniform(0, 2*np.pi, 16), rngf.uniform(-1, 1, 16),
                        rngf.uniform(0, np.pi, 16), np.full(16, lam_med), np.full(16, sb_med)])
                    aw16, ew16 = [], []
                    s0w = meta['c_bin'] - W // 2
                    mdl2 = ucb(f_lims=np.array([s0w*gen.df, (s0w+W)*gen.df]),
                               f_inds=[s0w, s0w+W], gpu=True, tobs=gen.t_obs, dt=gen.dt)
                    a_r, e_r, _ = mdl2.eval(P16)
                    a_r = cp.asnumpy(cp.asarray(a_r)); e_r = cp.asnumpy(cp.asarray(e_r))
                    sis = cp.asnumpy(cp.asarray(mdl2._template.start_inds)).ravel().astype(int)
                    K, L = a_r.shape
                    wpsd = cp.asnumpy(gen.sqrt_psd_g[s0w:s0w+W])
                    basis = np.zeros((K, 2*W), dtype=complex)
                    for k in range(K):
                        aa = np.zeros(W, complex); ee = np.zeros(W, complex)
                        lo2, hi2 = max(s0w, sis[k]), min(s0w+W, sis[k]+L)
                        if hi2 > lo2:
                            aa[lo2-s0w:hi2-s0w] = a_r[k, lo2-sis[k]:hi2-sis[k]]
                            ee[lo2-s0w:hi2-s0w] = e_r[k, lo2-sis[k]:hi2-sis[k]]
                        basis[k] = np.concatenate([aa/wpsd, ee/wpsd])
                    # PER-CHANNEL rank-4 fit (8 dof total): independent channel
                    # amplitudes absorb the sky-dependent relative response,
                    # removing the sky-mismatch floor (weak 1-month modulation).
                    fit = np.zeros(2*W, dtype=complex)
                    for ch in (slice(0, W), slice(W, 2*W)):
                        Uc, Sc_, Vhc = np.linalg.svd(basis[:, ch], full_matrices=False)
                        V4 = Vhc[:4]
                        fit[ch] = V4.T @ (V4.conj() @ d_vec[ch])
                    res = float(np.linalg.norm(d_vec - fit))
                    if res < best_res:
                        best_res, best_fit, best_dfb = res, fit, dfb
            aw_f = cp.asarray(best_fit[:W]); ew_f = cp.asarray(best_fit[W:])
        elif args.polish:
            # CEM polish on the WINDOW likelihood (observed data), seeded by flow
            xw = x.squeeze(0).cpu().numpy()
            dA_w = cp.asarray(xw[:, 0] + 1j * xw[:, 1])
            dE_w = cp.asarray(xw[:, 2] + 1j * xw[:, 3])
            s0w = meta['c_bin'] - W // 2
            mdl = ucb(f_lims=np.array([s0w * gen.df, (s0w + W) * gen.df]),
                      f_inds=[s0w, s0w + W], gpu=True, tobs=gen.t_obs, dt=gen.dt)
            wpsd = gen.sqrt_psd_g[s0w:s0w + W]

            def batch_lnl(P8b):
                a_r, e_r, _ = mdl.eval(P8b)
                a_r = cp.asarray(a_r); e_r = cp.asarray(e_r)
                sis = cp.asnumpy(cp.asarray(mdl._template.start_inds)).ravel().astype(int)
                K, L = a_r.shape
                out = np.full(K, -1e12)
                for k in range(K):
                    aw = cp.zeros(W, dtype=cp.complex128); ew = cp.zeros(W, dtype=cp.complex128)
                    lo, hi = max(s0w, sis[k]), min(s0w + W, sis[k] + L)
                    if hi > lo:
                        aw[lo - s0w:hi - s0w] = a_r[k, lo - sis[k]:hi - sis[k]]
                        ew[lo - s0w:hi - s0w] = e_r[k, lo - sis[k]:hi - sis[k]]
                    aw /= wpsd; ew /= wpsd
                    r = cp.sum(cp.abs(dA_w - aw)**2 + cp.abs(dE_w - ew)**2)
                    out[k] = -float(cp.asnumpy(r))
                return out

            mu = P8.mean(0); cov = np.cov(P8.T) + np.eye(8) * 1e-14
            cov[2, :] = 0; cov[:, 2] = 0; cov[2, 2] = 1e-20      # fdot fixed
            best_ll, best_p = -np.inf, P8[0]
            rngc = np.random.default_rng(500 + done)
            for it in range(10):
                pop = rngc.multivariate_normal(mu, cov, size=96)
                pop[:, 2] = FDOT_LOG10
                ll = batch_lnl(pop)
                o = np.argsort(ll)[::-1]
                if ll[o[0]] > best_ll:
                    best_ll, best_p = ll[o[0]], pop[o[0]]
                el = pop[o[:16]]
                mu = el.mean(0)
                cov = 0.7 * (np.cov(el.T) + np.eye(8) * 1e-16) + 0.3 * cov * 0.4
                cov[2, :] = 0; cov[:, 2] = 0; cov[2, 2] = 1e-20
            aw_f, ew_f = wf_window_batch(gen, best_p.reshape(1, 8), meta['c_bin'])
        else:
            aw_f, ew_f = wf_window_batch(gen, P8, meta['c_bin'])
        # true whitened signal (stored feats)
        aw_t = cp.asarray(feats[:, 0] + 1j * feats[:, 1])
        ew_t = cp.asarray(feats[:, 2] + 1j * feats[:, 3])
        rA = aw_t - aw_f; rE = ew_t - ew_f
        rsnr = float(cp.sqrt(4 * gen.df * cp.sum(cp.abs(rA)**2 + cp.abs(rE)**2)))
        # local alpha over the core (top-16 power bins of the true signal)
        pw = cp.asnumpy(cp.abs(aw_t)**2 + cp.abs(ew_t)**2)
        core = np.argsort(pw)[::-1][:16]
        a_loc = float(cp.asnumpy(cp.sqrt((cp.abs(rA)**2 + cp.abs(rE)**2)[cp.asarray(core)].mean() / 2)) / sigma0)
        res_snr.append(rsnr); alph.append(a_loc); src_snr.append(meta['snr'])
        done += 1
        if done % 50 == 0:
            print(f"  {done}/{args.n}", flush=True)

    res_snr = np.array(res_snr); alph = np.array(alph); src_snr = np.array(src_snr)
    print(f"\nsources: {args.n}, SNR median {np.median(src_snr):.0f}")
    print(f"residual SNR: median {np.median(res_snr):.2f}  p90 {np.percentile(res_snr, 90):.2f}"
          f"   (theory ~sqrt(7)~2.6 for optimal PE)")
    print(f"local alpha (core bins): median {np.median(alph):.2f}  p90 {np.percentile(alph, 90):.2f}"
          f"   (pointing budget: alpha <= 2)")
    for lo, hi in [(10, 50), (50, 200), (200, 1e9)]:
        sel = (src_snr >= lo) & (src_snr < hi)
        if sel.sum() > 5:
            print(f"  SNR[{lo:.0f},{hi:.0f}): residSNR {np.median(res_snr[sel]):.2f}, "
                  f"alpha {np.median(alph[sel]):.2f} (n={sel.sum()})")
    # global-equivalent contamination for a 40-source foreground:
    # each source leaves ~16 core bins at alpha_loc; over 39421 bins:
    a_glob = np.sqrt(40 * 16 * np.median(alph)**2 / 39421)
    print(f"global-equivalent alpha (40 sources): {a_glob:.3f}  << 2 budget")


if __name__ == '__main__':
    main()
