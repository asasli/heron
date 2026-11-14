"""
RECONSTRUCT-THEN-LOCALIZE feasibility test.

Question: if a wavelet/flow reconstruction (HyperWave-class, ~1 h) pins the
coherent waveform to fidelity alpha (template error = alpha x noise level,
RANDOM per bin — unbiased, unlike parametric ridge errors), how well does the
coherent sky fit localize?

Emulation: empirical template = true whitened h(f) + alpha*sigma0*randn (ONE
fixed draw per realization — it is one reconstruction), used in a two-stage
sky-grid coherent likelihood (intrinsics implicitly carried by the template;
the grid moves only the sky via the parametric response at true intrinsics,
so the test isolates the effect of reconstruction noise vs ridge noise).

Output per alpha: median 90% sky area + truth-in-90% over realizations.
Compare: alpha=0 (oracle, ~0.02 deg^2) vs parametric 0.1% Mc (2.4 deg shift,
coverage FAIL — the documented ridge pathology).

Run: python evaluation/recon_sky_test.py --case 1
"""
import sys, argparse, time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from heron.artifacts import (MBHB_FLOW, MBHB_NORM, MBHB_RECAL, UCB_FLOW, UCB_RESPONSE,
                             MBHB_DATA, UCB_DATA, load_kernel, product_out,
                             product_in, figure_out)
import cupy as cp
from pe_smbhb.data_mbhb import HetDataGenerator
from heron.sources import SANGRIA as CASES

ALPHAS = [0.0, 0.5, 1.0, 2.0]
N_REAL = 4


def sky_lnl_grid(gen, dA, dE, p_base, sb_c, lam_c, half_deg, n_side):
    """Coherent lnL over a sky grid centered at (sb_c, lam_c); template noise
    is ALREADY inside dA/dE handling (we perturb the template globally, see
    caller). Returns grids and lnL array. bbhx-safe: pads with prior rows."""
    beta_c = np.arcsin(np.clip(sb_c, -1, 1))
    dbet = np.deg2rad(half_deg)
    betas = beta_c + np.linspace(-dbet, dbet, n_side)
    lams = lam_c + np.linspace(-dbet, dbet, n_side) / max(np.cos(beta_c), 0.2)
    BB, LL = np.meshgrid(betas, lams, indexing='ij')
    P = np.tile(p_base, (BB.size, 1))
    P[:, 7] = np.sin(BB.ravel())
    P[:, 8] = LL.ravel() % (2 * np.pi)
    lnl = np.full(len(P), -1e12)
    half = 4 * gen.df
    for a0 in range(0, len(P), 32):
        rows = P[a0:a0 + 32]
        Ppad, _ = gen.draw_priors(64)
        Ppad[:len(rows)] = rows
        Aw, Ew = gen.gen_wf(Ppad)
        hA = Aw[:len(rows)] / gen.sqrt_psd_g[None, :]
        hE = Ew[:len(rows)] / gen.sqrt_psd_g[None, :]
        r = cp.abs(hA - dA[None, :])**2 + cp.abs(hE - dE[None, :])**2
        lnl[a0:a0 + len(rows)] = cp.asnumpy(-0.5 * half * cp.sum(r, axis=1))
    return BB, LL, lnl.reshape(n_side, n_side)


def area_and_cover(BB, LL, lnl, sb_t, lam_t):
    """90% credible area [deg^2] + truth-in-90% from a lnL grid."""
    p = np.exp(lnl - lnl.max())
    p /= p.sum()
    dbet = BB[1, 0] - BB[0, 0]
    dlam = LL[0, 1] - LL[0, 0]
    pix = np.rad2deg(dbet) * np.rad2deg(dlam) * np.cos(BB)  # deg^2 per pixel
    order = np.argsort(p.ravel())[::-1]
    csum = np.cumsum(p.ravel()[order])
    in90 = order[:np.searchsorted(csum, 0.90) + 1]
    area = pix.ravel()[in90].sum()
    # truth pixel inside the 90% set?
    ib = np.argmin(np.abs(BB[:, 0] - np.arcsin(sb_t)))
    il = np.argmin(np.abs(((LL[0, :] - lam_t + np.pi) % (2*np.pi)) - np.pi))
    truth_flat = ib * BB.shape[1] + il
    cov = truth_flat in set(in90.tolist())
    return area, cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', type=int, default=1, choices=[1, 2])
    ap.add_argument('--t-pre', type=float, default=24.0)
    args = ap.parse_args()
    S = CASES[args.case]['src']; CNAME = CASES[args.case]['name']

    gen = HetDataGenerator(output_dir=None)
    np.random.seed(123)
    for _ in range(2):
        Pw, _ = gen.draw_priors(64); gen.gen_wf(Pw)
    m1, m2 = S['m1'], S['m2']; eta = m1*m2/(m1+m2)**2
    Mc = (m1+m2)*eta**0.6; q = m1/m2
    t_ref = (gen.t_obs - args.t_pre*3600)/3600
    sb_t, lam_t = np.sin(S['beta']), S['lam']
    p_true = np.array([Mc, q, S['chi1'], S['chi2'], np.log10(S['dist_Mpc']), t_ref,
                       np.cos(S['iota']), sb_t, lam_t, S['psi'], S['phi0']])
    A, E = gen.gen_wf(p_true[None, :])
    hA_t = (A[0]/gen.sqrt_psd_g).astype(cp.complex64)
    hE_t = (E[0]/gen.sqrt_psd_g).astype(cp.complex64)
    snr = float(cp.sqrt(4*gen.df*cp.sum(cp.abs(hA_t)**2 + cp.abs(hE_t)**2)))
    print(f"{CNAME}: SNR={snr:.0f} | recon-fidelity sky test "
          f"(alpha = template error / noise level)")

    rng0 = cp.random.RandomState(7)
    nA = (rng0.standard_normal(gen.n_bins) + 1j*rng0.standard_normal(gen.n_bins)) * gen.sigma0
    nE = (rng0.standard_normal(gen.n_bins) + 1j*rng0.standard_normal(gen.n_bins)) * gen.sigma0
    dA_data = hA_t + nA.astype(cp.complex64)   # the observed data (fixed)
    dE_data = hE_t + nE.astype(cp.complex64)

    print(f"{'alpha':>6} {'real':>4} | {'area90 [deg^2]':>14} {'truth in90':>10} | {'peak shift[deg]':>15}")
    for alpha in ALPHAS:
        for r in range(N_REAL if alpha > 0 else 1):
            rng = cp.random.RandomState(100 + r + int(alpha*10)*10)
            # empirical template: truth + reconstruction noise (random, unbiased)
            eA = (rng.standard_normal(gen.n_bins) + 1j*rng.standard_normal(gen.n_bins)) * gen.sigma0 * alpha
            eE = (rng.standard_normal(gen.n_bins) + 1j*rng.standard_normal(gen.n_bins)) * gen.sigma0 * alpha
            # "moving" the empirical template in sky == moving the parametric
            # part; the recon noise rides along as a fixed additive perturbation
            # of the DATA side (equivalent algebra: |h(sky)+e - d|^2)
            dA_eff = dA_data - eA.astype(cp.complex64)
            dE_eff = dE_data - eE.astype(cp.complex64)
            t0 = time.time()
            # stage 1: +-4 deg @ 33x33 -> stage 2: zoom +-0.5 deg @ 41x41
            BB, LL, lnl = sky_lnl_grid(gen, dA_eff, dE_eff, p_true, sb_t, lam_t, 4.0, 33)
            lnl = lnl / (1.0 + alpha**2)          # marginalize recon error
            i, j = np.unravel_index(np.argmax(lnl), lnl.shape)
            sb_pk, lam_pk = np.sin(BB[i, j]), LL[i, j]
            BB2, LL2, lnl2 = sky_lnl_grid(gen, dA_eff, dE_eff, p_true,
                                          sb_pk, lam_pk, 0.5, 41)
            lnl2 = lnl2 / (1.0 + alpha**2)        # marginalize recon error
            area, cov = area_and_cover(BB2, LL2, lnl2, sb_t, lam_t)
            i2, j2 = np.unravel_index(np.argmax(lnl2), lnl2.shape)
            shift = np.rad2deg(np.arccos(np.clip(
                np.sin(BB2[i2, j2])*sb_t + np.cos(BB2[i2, j2])*np.cos(np.arcsin(sb_t))
                * np.cos(LL2[i2, j2]-lam_t), -1, 1)))
            print(f"{alpha:>6.1f} {r:>4} | {area:>14.3f} {str(cov):>10} | {shift:>15.3f}"
                  f"   ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
