"""
Heterodyned + binned 1-month LISA SMBHB dataset for sky-capable NPE.

Why: the decimated raw-FD representation (data_v2_1month) aliases the chirp
phase (~7 cycles between stored bins), destroying the Doppler/transfer-function
phase that carries the sky information. A full-resolution check shows that the
full-res coherent likelihood localizes to ~0.2 deg^2 at SNR 300, and 4096
heterodyned bin-averaged summaries preserve that even with an imperfect
reference (Dingo-BNS-style relative binning).

Per sample:
  1. Draw theta_true from the same prior as generate_yearly_v2 (SNR >= MIN_SNR).
  2. Draw theta_ref = theta_true perturbed in (Mc, q, c1, c2, t_ref) by a
     narrow (80%) or wide (20%) kernel; extrinsics fixed to fiducial
     (ecliptic pole, face-on) so |A_ref| has no antenna nulls.
  3. Heterodyne: c(f) = conj(A_ref)/|A_ref|; bin-average whitened A,E into
     N_BINS uniform bins.
  4. Store (N_BINS, 5) float32: [rA.re, rA.im, rE.re, rE.im, |A_ref|_whitened]
     signal-only; Gaussian noise with per-bin std sigma0/sqrt(n_k) is added
     on the fly during training (sigma0 = 1/(2 sqrt(df)) in whitened units).

Run:

    python pe_smbhb/data_mbhb.py --output-dir data/mbhb --wide-spin-ref \
        --n-train 200000 --n-val 5000 --n-test 2000 --seed 90000
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import *

sys.path.insert(0, NGB_PATH)
from data_utils import smbhb, noise_model

import cupy as cp

N_BINS_OUT = 4096

# Reference-perturbation kernel (must match inference-time GNPE kernel).
# v3: a *ladder* of kernel scales. Each sample draws a scale s multiplying the
# base widths: 15% s=4 (wide; covers stage-1 entry points), else
# s ~ logU[0.02, 1]. The network is conditioned on log10(s), enabling
# GNPE-style iteration at inference: each round's tighter reference is
# in-distribution, residual phases shrink, and the sky phase becomes readable.
# Motivation: sky area correlates with the reference TIME error (rho=0.41),
# not the Mc error — the 120 s time floor was the sky bottleneck.
KERNEL = {
    'base': {'frac_Mc': 0.003, 'log_q': 0.05, 'chi': 0.05,
             't_floor_sec': 120.0, 't_frac_of_tpre': 0.02},
    's_wide': 4.0,
    'p_wide': 0.15,
    's_min': 0.02,
    # ridge kernel (v6): references whose (q, chi) errors are O(base) but
    # whose (Mc, t) are phase-compensated along the degeneracy ridge — the
    # error pattern produced by likelihood (matched-filter) reference
    # refinement at inference. s_phase = residual phase-coherence scale;
    # the parameter-space spread is ridge-correlated and ~s_eff in base units.
    'p_ridge': 0.5,
    'ridge_s_qc': (0.25, 1.5),       # logU range for (q, chi) jitter scale
    'ridge_s_phase': (0.02, 0.3),    # logU range for residual coherence
    'ridge_s_eff_coef': 3.0,         # residual scale = coef * s_qc (base units)
}
FIDUCIAL_EXT = {'logD': 4.0, 'ci': 1.0, 'sb': 0.999, 'lam': 0.0,
                'psi': 0.0, 'phi0': 0.0}
# v15: reference spins drawn UNINFORMATIVELY (U[-0.99,0.99], independent of
# truth) so the network must extract chi from the data instead of echoing the
# truth-centered reference (the reference-anchored spin pathology). Safe per
# scripts/probe_spin_het.py (SNR retention 0.9999 at chi_err~1, no ridge
# compensation needed). Enable with env WIDE_SPIN_REF=1.
import os
WIDE_SPIN_REF = bool(int(os.environ.get('WIDE_SPIN_REF', '0')))


class HetDataGenerator:
    def __init__(self, output_dir=None, t_obs_years=1.0/12.0,
                 f_min=1e-5, f_max=1.5e-2, no_confusion=False, batch=64):
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.batch = batch
        self.t_obs = t_obs_years * YRSID_SI
        self.dt = 1.0
        self.df = 1.0 / self.t_obs
        ndata = int(self.t_obs / self.dt)
        nfft = int(ndata / 2 + 1) if ndata % 2 == 0 else int((ndata + 1) / 2)
        F = self.df * nfft
        fvec_full = np.arange(0, F, self.df)
        f_inds = np.where((fvec_full >= f_min) & (fvec_full <= f_max))[0]
        self.freqs = fvec_full[f_inds]
        self.n_bins = len(self.freqs)

        nm = noise_model(self.freqs)
        psd_instr = nm.eval([SA, SI])
        psd_instr = np.asarray(psd_instr.get() if hasattr(psd_instr, 'get')
                               else psd_instr).ravel()
        if no_confusion:
            psd = psd_instr.copy()
        else:
            psd = psd_instr + self._galactic_confusion_psd()
        psd[psd <= 0] = np.inf
        self.psd_g = cp.asarray(psd)
        self.sqrt_psd_g = cp.sqrt(self.psd_g)

        # Uniform binning of the full-res grid into N_BINS_OUT bins
        self.edges = np.linspace(0, self.n_bins, N_BINS_OUT + 1).astype(int)
        self.counts = np.diff(self.edges)
        self.counts_g = cp.asarray(self.counts.astype(np.float64))
        self.sigma0 = 1.0 / (2.0 * np.sqrt(self.df))  # whitened per-Re/Im noise std

        self.template = smbhb(f=self.freqs, modes=[(2, 2)],
                              tstart=self.t_obs, tend=0.0, gpu=True)

        print(f"Heterodyned 1-month generator")
        print(f"  T_obs={self.t_obs/86400:.2f} d  df={self.df:.3e}  "
              f"full bins={self.n_bins} -> {N_BINS_OUT} binned")
        print(f"  sigma0 (full-res whitened noise std) = {self.sigma0:.1f}")

    def _galactic_confusion_psd(self):
        f = self.freqs
        Sc = 9e-45 * f**(-7/3) * np.exp(-f**0.138 - 221 * f * np.sin(521 * f))
        Sc *= (1 + np.tanh(1680 * (0.00113 - f)))
        Sc *= (2.0 * np.pi * f / 299792458.0)**2
        return Sc

    def bin_mean(self, X):
        """Segmented mean over axis 1 via cumsum. X: (B, n_bins) complex."""
        S = cp.cumsum(X, axis=1)
        upper = S[:, self.edges[1:] - 1]
        lower = cp.zeros_like(upper)
        nz = self.edges[:-1] > 0
        lower[:, nz] = S[:, self.edges[:-1][nz] - 1]
        return (upper - lower) / self.counts_g[None, :]

    def draw_priors(self, B):
        m1 = np.random.uniform(SMBHB_M1_MIN, SMBHB_M1_MAX, B)
        m2 = np.random.uniform(SMBHB_M2_MIN, SMBHB_M2_MAX, B)
        m1, m2 = np.maximum(m1, m2), np.minimum(m1, m2)
        eta = m1 * m2 / (m1 + m2)**2
        Mc = (m1 + m2) * eta**0.6
        q = m1 / m2
        dist = np.random.uniform(SMBHB_DIST_MIN, SMBHB_DIST_MAX, B)
        logD = np.log10(dist)
        u = np.random.random(B)
        t_pre_hours = np.where(
            u < 0.7,
            10**np.random.uniform(np.log10(1/60), np.log10(24*25), B),
            np.random.uniform(0.5, 5.0, B))
        t_ref_hr = (self.t_obs - t_pre_hours * 3600.0) / 3600.0
        P = np.stack([
            Mc, q,
            np.random.uniform(-1, 1, B), np.random.uniform(-1, 1, B),  # c1, c2
            logD, t_ref_hr,
            np.random.uniform(-1, 1, B),                               # cos_inc
            np.random.uniform(-1, 1, B),                               # sin_beta
            np.random.uniform(0, 2*np.pi, B),                          # lambda
            np.random.uniform(0, np.pi, B),                            # psi
            np.random.uniform(0, 2*np.pi, B),                          # phi0
        ], axis=1)
        return P, t_pre_hours

    def draw_refs(self, P_true, t_pre_hours):
        """Perturbed reference params: intrinsic+t jittered by a per-sample
        kernel scale s (ladder), extrinsics fiducial."""
        B = P_true.shape[0]
        wide = np.random.random(B) < KERNEL['p_wide']
        s = np.where(wide, KERNEL['s_wide'],
                     10**np.random.uniform(np.log10(KERNEL['s_min']), 0.0, B))
        k = KERNEL['base']
        P = P_true.copy()
        P[:, 0] *= 1.0 + np.random.randn(B) * k['frac_Mc'] * s
        P[:, 1] = np.maximum(P[:, 1] * np.exp(np.random.randn(B) * k['log_q'] * s), 1.0001)
        P[:, 2] = np.clip(P[:, 2] + np.random.randn(B) * k['chi'] * s, -0.999, 0.999)
        P[:, 3] = np.clip(P[:, 3] + np.random.randn(B) * k['chi'] * s, -0.999, 0.999)
        sig_t = np.maximum(k['t_floor_sec'],
                           k['t_frac_of_tpre'] * t_pre_hours * 3600.0) * s
        P[:, 5] += np.random.randn(B) * sig_t / 3600.0
        P[:, 4] = FIDUCIAL_EXT['logD']
        P[:, 6] = FIDUCIAL_EXT['ci']
        P[:, 7] = FIDUCIAL_EXT['sb']
        P[:, 8] = FIDUCIAL_EXT['lam']
        P[:, 9] = FIDUCIAL_EXT['psi']
        P[:, 10] = FIDUCIAL_EXT['phi0']
        return P, s

    def gen_wf(self, P):
        A, E, _ = self.template.eval(P)
        A = cp.asarray(A).reshape(P.shape[0], -1)
        E = cp.asarray(E).reshape(P.shape[0], -1)
        return A, E

    def ridge_compensate(self, P_true, P_pert, t_pre_hours):
        """For (q, chi)-perturbed references, find the (dMc, dt) that
        phase-compensates the perturbation (weighted LS on the unwrapped
        phase difference; basis = 2*pi*f and the numerical dPsi/dMc).
        Mimics what likelihood refinement produces at inference.
        Returns (dMc, dt_sec) arrays."""
        B = P_true.shape[0]
        P_fid = P_true.copy()                 # fiducial-extrinsic, true intrinsics
        for col, key in [(4, 'logD'), (6, 'ci'), (7, 'sb'), (8, 'lam'),
                         (9, 'psi'), (10, 'phi0')]:
            P_fid[:, col] = FIDUCIAL_EXT[key]
        P_mc = P_fid.copy()
        delta = 1e-3
        P_mc[:, 0] *= (1 + delta)

        A_fid, _ = self.gen_wf(P_fid)
        A_pert, _ = self.gen_wf(P_pert)
        A_mc, _ = self.gen_wf(P_mc)

        w = (cp.abs(A_fid)**2 / self.psd_g[None, :])
        live = (cp.abs(A_fid) > 0) & (cp.abs(A_pert) > 0) & (cp.abs(A_mc) > 0)
        w = w * live

        def gated_unwrap(X):
            """Accumulate principal-value phase increments over live bins only
            (np.unwrap through zero-amplitude regions injects garbage that
            corrupts everything downstream of the gap)."""
            inc = cp.angle(X[:, 1:] * cp.conj(X[:, :-1]))
            inc = inc * (live[:, 1:] & live[:, :-1])
            out = cp.zeros(X.shape, dtype=cp.float64)
            out[:, 1:] = cp.cumsum(inc, axis=1)
            return out

        dpsi = gated_unwrap(A_pert * cp.conj(A_fid))
        b_mc = gated_unwrap(A_mc * cp.conj(A_fid)) \
            / (cp.asarray(P_true[:, 0])[:, None] * delta)        # dPsi/dMc
        b_t = (2 * np.pi * cp.asarray(self.freqs)[None, :]
               * cp.ones((B, 1)))
        b_0 = cp.ones_like(b_t)                                  # phi0 freedom

        # per-sample 3x3 weighted LS: minimize
        # sum w (dpsi + c0 + b_t*dt + b_mc*dMc)^2 -> ref absorbs perturbation
        basis = cp.stack([b_0, b_t, b_mc], axis=1)               # (B, 3, F)
        wb = basis * w[:, None, :]
        A_mat = wb @ basis.transpose(0, 2, 1)                    # (B, 3, 3)
        rhs = -(wb @ dpsi[:, :, None])                           # (B, 3, 1)
        A_mat += 1e-12 * cp.eye(3)[None, :, :]
        sol = cp.linalg.solve(A_mat, rhs)[:, :, 0]
        dt = cp.asnumpy(sol[:, 1])
        dmc = cp.asnumpy(sol[:, 2])
        return dmc, dt

    def draw_refs_ridge(self, P_true, t_pre_hours):
        """Ridge-kernel references (v6): see KERNEL comment."""
        B = P_true.shape[0]
        k = KERNEL['base']
        lo, hi = KERNEL['ridge_s_qc']
        s_qc = 10**np.random.uniform(np.log10(lo), np.log10(hi), B)
        lo, hi = KERNEL['ridge_s_phase']
        s_phase = 10**np.random.uniform(np.log10(lo), np.log10(hi), B)
        P = P_true.copy()
        P[:, 1] = np.maximum(P[:, 1] * np.exp(np.random.randn(B) * k['log_q'] * s_qc), 1.005)
        P[:, 2] = np.clip(P[:, 2] + np.random.randn(B) * k['chi'] * s_qc, -0.99, 0.99)
        P[:, 3] = np.clip(P[:, 3] + np.random.randn(B) * k['chi'] * s_qc, -0.99, 0.99)
        for col, key in [(4, 'logD'), (6, 'ci'), (7, 'sb'), (8, 'lam'),
                         (9, 'psi'), (10, 'phi0')]:
            P[:, col] = FIDUCIAL_EXT[key]
        dmc, dt = self.ridge_compensate(P_true, P, t_pre_hours)
        sig_t = np.maximum(k['t_floor_sec'],
                           k['t_frac_of_tpre'] * t_pre_hours * 3600.0)
        P[:, 0] = P_true[:, 0] + dmc \
            + np.random.randn(B) * k['frac_Mc'] * s_phase * P_true[:, 0]
        P[:, 5] = P_true[:, 5] + dt / 3600.0 \
            + np.random.randn(B) * sig_t * s_phase / 3600.0
        return P, s_phase, s_qc

    def process_batch(self, P_true, t_pre_hours):
        """Returns (B, N_BINS_OUT, 5) float32 features, snr (B,), P_ref, meta."""
        A, E = self.gen_wf(P_true)
        wA = A / self.sqrt_psd_g[None, :]
        wE = E / self.sqrt_psd_g[None, :]
        snr = cp.asnumpy(cp.sqrt(
            4 * self.df * cp.sum(cp.abs(wA)**2 + cp.abs(wE)**2, axis=1)))

        # split batch into independent-kernel and ridge-kernel references
        B = P_true.shape[0]
        is_ridge = np.random.random(B) < KERNEL['p_ridge']
        P_ref, kscale = self.draw_refs(P_true, t_pre_hours)
        s_phase = kscale.copy()             # indep: phase scale == param scale
        if is_ridge.any():
            Pr, spr, sqc = self.draw_refs_ridge(P_true[is_ridge],
                                                t_pre_hours[is_ridge])
            P_ref[is_ridge] = Pr
            s_phase[is_ridge] = spr
            kscale[is_ridge] = np.clip(KERNEL['ridge_s_eff_coef'] * sqc, 0.5, 4.0)
        if WIDE_SPIN_REF:
            P_ref[:, 2] = np.random.uniform(-0.99, 0.99, B)
            P_ref[:, 3] = np.random.uniform(-0.99, 0.99, B)
        Aref, _ = self.gen_wf(P_ref)
        mag = cp.abs(Aref)
        c = cp.where(mag > 0, cp.conj(Aref) / cp.where(mag > 0, mag, 1.0), 0.0)

        rA = self.bin_mean(wA * c)
        rE = self.bin_mean(wE * c)
        href = self.bin_mean((mag / self.sqrt_psd_g[None, :]).astype(cp.complex128)).real

        feats = cp.stack([rA.real, rA.imag, rE.real, rE.imag, href],
                         axis=-1).astype(cp.float32)
        return cp.asnumpy(feats), snr, P_ref, kscale, s_phase, is_ridge

    def generate(self, n_train, n_val, n_test, seed=SEED):
        np.random.seed(seed)
        # global (sample-independent) arrays
        self.output_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.output_dir / 'bin_counts.npy', self.counts)
        centers = np.array([self.freqs[a:b].mean()
                            for a, b in zip(self.edges[:-1], self.edges[1:])])
        np.save(self.output_dir / 'bin_freqs.npy', centers)
        info = {'df': self.df, 't_obs': self.t_obs, 'n_bins_full': self.n_bins,
                'n_bins_out': N_BINS_OUT, 'sigma0': self.sigma0,
                'kernel': KERNEL, 'fiducial_ext': FIDUCIAL_EXT,
                'wide_spin_ref': WIDE_SPIN_REF, 'seed': int(seed),
                'n_train': int(n_train), 'n_val': int(n_val), 'n_test': int(n_test),
                'param_order': ['Mc', 'q', 'c1', 'c2', 'logD', 't_ref_hr',
                                'cos_inc', 'sin_beta', 'lambda', 'psi', 'phi0']}
        (self.output_dir / 'dataset_info.json').write_text(json.dumps(info, indent=2))

        for split, n_samples in [('train', n_train), ('val', n_val), ('test', n_test)]:
            if n_samples == 0:
                continue
            split_dir = self.output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            metadata = []
            pbar = tqdm(total=n_samples, desc=split)
            saved = 0
            while saved < n_samples:
                B = min(self.batch, 2 * (n_samples - saved) + 8)
                P_true, t_pre = self.draw_priors(B)
                feats, snr, P_ref, kscale, s_phase, is_ridge = \
                    self.process_batch(P_true, t_pre)
                keep = np.where(snr >= MIN_SNR)[0]
                for b in keep:
                    if saved >= n_samples:
                        break
                    np.save(split_dir / f'{saved:06d}_het.npy', feats[b])
                    pt, pr = P_true[b], P_ref[b]
                    metadata.append({
                        'idx': saved, 'has_smbhb': True,
                        'smbhb': {
                            'Mc': float(pt[0]), 'q': float(pt[1]),
                            'c1': float(pt[2]), 'c2': float(pt[3]),
                            'logD': float(pt[4]), 'cos_inc': float(pt[6]),
                            'sin_beta': float(pt[7]), 'lambda': float(pt[8]),
                            'psi': float(pt[9]), 'phi0': float(pt[10]),
                            't_pre_hours': float(t_pre[b]),
                            'snr': float(snr[b]),
                        },
                        'ref': {
                            'Mc': float(pr[0]), 'q': float(pr[1]),
                            'c1': float(pr[2]), 'c2': float(pr[3]),
                            't_ref_hr': float(pr[5]),
                            'kernel_scale': float(kscale[b]),
                            's_phase': float(s_phase[b]),
                            'kernel_type': int(is_ridge[b]),
                        },
                    })
                    saved += 1
                    pbar.update(1)
            pbar.close()
            with open(split_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f)
            snrs = [m['smbhb']['snr'] for m in metadata]
            print(f"  {split}: {saved} samples, SNR [{min(snrs):.0f}, "
                  f"{max(snrs):.0f}], median {np.median(snrs):.0f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='data/mbhb')
    parser.add_argument('--wide-spin-ref', action='store_true',
                        help='draw reference spins uninformatively from U[-0.99, 0.99] '
                             '(the published model; same as env WIDE_SPIN_REF=1)')
    parser.add_argument('--n-train', type=int, default=200000)
    parser.add_argument('--n-val', type=int, default=5000)
    parser.add_argument('--n-test', type=int, default=2000)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--no-confusion', action='store_true')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    if args.wide_spin_ref:
        WIDE_SPIN_REF = True

    gen = HetDataGenerator(output_dir=args.output_dir, batch=args.batch,
                           no_confusion=args.no_confusion)
    gen.generate(args.n_train, args.n_val, args.n_test, seed=args.seed)
