"""
Robust heterodyned dataset: SMBHB + resolved UCBs + varying galactic-confusion
(SGWB-from-UCBs) foreground. Extends generate_het_1month so the sky-GNPE model
can be conditioned on n_ucb (the count YOLO provides) and trained across many
confusion realizations/amplitudes -> robust to the galactic foreground.

Per sample, data = SMBHB(true) + sum of n_ucb resolved UCBs
                    + confusion realization (amplitude conf_amp x nominal),
then whitened by the nominal (instrument+confusion) PSD and heterodyned by the
SMBHB reference (exactly as the clean pipeline). Instrument noise is still added
on the fly in training. Stores n_ucb and conf_amp in metadata.

Run:
     python -m pe_smbhb.data_mbhb_ucb \
        --output-dir data/data_het_ucb --n-train 200000 \
        --n-val 5000 --n-test 2000 --seed 70
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
from data_utils import ucb

import cupy as cp
from pe_smbhb.data_mbhb import (
    HetDataGenerator, KERNEL, FIDUCIAL_EXT, N_BINS_OUT)

# UCB population (resolved): count and frequency/amplitude ranges. n_ucb varies
# per sample so the model learns to condition on it; conf_amp varies the
# unresolved foreground so the model is robust to SGWB-level uncertainty.
UCB_N_MAX = 40
UCB_F_LO_MHZ, UCB_F_HI_MHZ = 0.3, 8.0      # galactic band
UCB_LOG10A = (-21.5, -20.3)                # resolved (bright) amplitudes
UCB_HALFWIDTH_HZ = 3e-6
CONF_AMP_RANGE = (0.5, 2.0)                 # confusion foreground scale ladder


class UCBHetGenerator(HetDataGenerator):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        # nominal confusion PSD (already in self.psd via parent); store the
        # confusion-only part for drawing realizations.
        f = self.freqs
        Sc = 9e-45 * f**(-7/3) * np.exp(-f**0.138 - 221 * f * np.sin(521 * f))
        Sc *= (1 + np.tanh(1680 * (0.00113 - f)))
        Sc *= (2.0 * np.pi * f / 299792458.0)**2
        self.Sc = cp.asarray(np.maximum(Sc, 0.0))
        self.conf_sigma = cp.sqrt(self.Sc / (2.0 * self.df))  # raw-space std

    def _gen_ucbs(self, n):
        """Sum of n resolved UCB waveforms (raw A/E, full-res cupy)."""
        A = cp.zeros(self.n_bins, dtype=cp.complex128)
        E = cp.zeros(self.n_bins, dtype=cp.complex128)
        for _ in range(n):
            f0 = np.random.uniform(UCB_F_LO_MHZ, UCB_F_HI_MHZ) * 1e-3
            hw = UCB_HALFWIDTH_HZ
            f_lims = np.array([f0 - hw, f0 + hw])
            s0 = int(f_lims[0] / self.df)
            s1 = int(f_lims[1] / self.df)
            if s0 < 1 or s1 >= self.n_bins:
                continue
            try:
                m = ucb(f_lims=f_lims, f_inds=[s0, s1 + 1], gpu=True,
                        tobs=self.t_obs, dt=self.dt)
                p = np.array([np.random.uniform(*UCB_LOG10A), f0 * 1e3, -16.5,
                              np.random.uniform(0, 2*np.pi), np.random.uniform(-1, 1),
                              np.random.uniform(0, np.pi), np.random.uniform(0, 2*np.pi),
                              np.random.uniform(-1, 1)])
                a, e, _ = m.eval(p.reshape(1, 8))
                a = cp.asarray(a).ravel(); e = cp.asarray(e).ravel()
                L = len(a)
                A[s0:s0+L] += a
                E[s0:s0+L] += e
            except Exception:
                continue
        return A, E

    def process_batch_ucb(self, P_true, t_pre_hours):
        """As parent process_batch but injects UCBs + confusion before
        heterodyning. Returns feats, snr, P_ref, kscale, s_phase, is_ridge,
        n_ucb, conf_amp."""
        A, E = self.gen_wf(P_true)                      # raw SMBHB
        B = P_true.shape[0]
        # SMBHB SNR (signal only, nominal PSD)
        wA0 = A / self.sqrt_psd_g[None, :]
        wE0 = E / self.sqrt_psd_g[None, :]
        snr = cp.asnumpy(cp.sqrt(4 * self.df * cp.sum(
            cp.abs(wA0)**2 + cp.abs(wE0)**2, axis=1)))

        n_ucb = np.random.randint(0, UCB_N_MAX + 1, size=B)
        conf_amp = 10 ** np.random.uniform(np.log10(CONF_AMP_RANGE[0]),
                                           np.log10(CONF_AMP_RANGE[1]), size=B)
        A_tot = A.copy(); E_tot = E.copy()
        for b in range(B):
            if n_ucb[b] > 0:
                uA, uE = self._gen_ucbs(int(n_ucb[b]))
                A_tot[b] += uA; E_tot[b] += uE
            # confusion realization (raw), amplitude-scaled
            cr_A = self.conf_sigma * (cp.random.standard_normal(self.n_bins)
                                      + 1j * cp.random.standard_normal(self.n_bins))
            cr_E = self.conf_sigma * (cp.random.standard_normal(self.n_bins)
                                      + 1j * cp.random.standard_normal(self.n_bins))
            A_tot[b] += conf_amp[b] * cr_A
            E_tot[b] += conf_amp[b] * cr_E

        wA = A_tot / self.sqrt_psd_g[None, :]
        wE = E_tot / self.sqrt_psd_g[None, :]

        # reference (identical scheme to parent)
        is_ridge = np.random.random(B) < KERNEL['p_ridge']
        P_ref, kscale = self.draw_refs(P_true, t_pre_hours)
        s_phase = kscale.copy()
        if is_ridge.any():
            Pr, spr, sqc = self.draw_refs_ridge(P_true[is_ridge], t_pre_hours[is_ridge])
            P_ref[is_ridge] = Pr
            s_phase[is_ridge] = spr
            kscale[is_ridge] = np.clip(KERNEL['ridge_s_eff_coef'] * sqc, 0.5, 4.0)
        Aref, _ = self.gen_wf(P_ref)
        mag = cp.abs(Aref)
        c = cp.where(mag > 0, cp.conj(Aref) / cp.where(mag > 0, mag, 1.0), 0.0)
        rA = self.bin_mean(wA * c)
        rE = self.bin_mean(wE * c)
        href = self.bin_mean((mag / self.sqrt_psd_g[None, :]).astype(cp.complex128)).real
        feats = cp.stack([rA.real, rA.imag, rE.real, rE.imag, href],
                         axis=-1).astype(cp.float32)
        return (cp.asnumpy(feats), snr, P_ref, kscale, s_phase, is_ridge,
                n_ucb, conf_amp)

    def generate(self, n_train, n_val, n_test, seed=SEED):
        np.random.seed(seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.output_dir / 'bin_counts.npy', self.counts)
        centers = np.array([self.freqs[a:b].mean()
                            for a, b in zip(self.edges[:-1], self.edges[1:])])
        np.save(self.output_dir / 'bin_freqs.npy', centers)
        info = {'df': self.df, 't_obs': self.t_obs, 'n_bins_full': self.n_bins,
                'n_bins_out': N_BINS_OUT, 'sigma0': self.sigma0,
                'kernel': KERNEL, 'fiducial_ext': FIDUCIAL_EXT,
                'ucb': {'n_max': UCB_N_MAX, 'f_mhz': [UCB_F_LO_MHZ, UCB_F_HI_MHZ],
                        'log10A': list(UCB_LOG10A), 'conf_amp': list(CONF_AMP_RANGE)},
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
                (feats, snr, P_ref, kscale, s_phase, is_ridge,
                 n_ucb, conf_amp) = self.process_batch_ucb(P_true, t_pre)
                keep = np.where(snr >= MIN_SNR)[0]
                for b in keep:
                    if saved >= n_samples:
                        break
                    np.save(split_dir / f'{saved:06d}_het.npy', feats[b])
                    pt, pr = P_true[b], P_ref[b]
                    metadata.append({
                        'idx': saved, 'has_smbhb': True,
                        'n_ucb': int(n_ucb[b]), 'conf_amp': float(conf_amp[b]),
                        'smbhb': {
                            'Mc': float(pt[0]), 'q': float(pt[1]),
                            'c1': float(pt[2]), 'c2': float(pt[3]),
                            'logD': float(pt[4]), 'cos_inc': float(pt[6]),
                            'sin_beta': float(pt[7]), 'lambda': float(pt[8]),
                            'psi': float(pt[9]), 'phi0': float(pt[10]),
                            't_pre_hours': float(t_pre[b]), 'snr': float(snr[b]),
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
            nu = [m['n_ucb'] for m in metadata]
            print(f"  {split}: {saved} samples, n_ucb [{min(nu)},{max(nu)}] "
                  f"median {int(np.median(nu))}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', default='data/data_het_ucb')
    ap.add_argument('--n-train', type=int, default=200000)
    ap.add_argument('--n-val', type=int, default=5000)
    ap.add_argument('--n-test', type=int, default=2000)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--seed', type=int, default=70)
    args = ap.parse_args()
    gen = UCBHetGenerator(output_dir=args.output_dir, batch=args.batch)
    gen.generate(args.n_train, args.n_val, args.n_test, seed=args.seed)
