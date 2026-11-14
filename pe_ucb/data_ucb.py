"""
UCB per-source PE dataset — the galactic-binary half of the HERON pipeline.

Each sample = ONE resolved UCB in a narrow frequency window around its
YOLO-detected frequency (the detection->PE hand-off):
  features: (W, 4) = whitened A/E (re, im) full-resolution bins, W=128
  targets:  7 params [log10A, df_bins, phi0, cos_iota, psi, lambda, sin_beta]
            (fdot is unmeasurable in 1 month; fixed to the detection value)

df_bins = (f_true - f_window_center) in full-res bins: YOLO centers the window
with a detection error (U[-8,+8] bins), the flow measures the fine frequency.

Stored: SIGNAL-ONLY window (complex -> re/im), so training adds on the fly:
  - instrument noise (whitened => unit std per bin)
  - confusion floor: extra Gaussian noise x sqrt(Sc/Sn)(f_win) x conf_amp ladder
  - NEIGHBOR sources: shifted/scaled windows of OTHER samples (zero-cost overlap
    robustness — the global-fit problem in miniature)

Run: python -m pe_ucb.data_ucb \
        --output-dir data/ucb --n-train 1000000
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
from pe_smbhb.data_mbhb import HetDataGenerator

W = 128                                   # window, full-res bins
UCB_F_LO_MHZ, UCB_F_HI_MHZ = 0.3, 8.0     # galactic band (matches detection)
UCB_LOG10A = (-22.0, -20.3)               # slightly deeper than detection floor
FDOT_LOG10 = -16.5                        # fixed (unmeasurable in 1 month)
DET_ERR_BINS = 8                          # YOLO center error, U[-8, 8] bins
PARAM_ORDER = ['log10A', 'df_bins', 'phi0', 'cos_iota', 'psi', 'lambda', 'sin_beta']


class UCBPEGenerator(HetDataGenerator):
    """Reuses HetDataGenerator's grid/PSD; generates single-UCB windows."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        f = self.freqs
        Sc = 9e-45 * f**(-7/3) * np.exp(-f**0.138 - 221 * f * np.sin(521 * f))
        Sc *= (1 + np.tanh(1680 * (0.00113 - f)))
        Sc *= (2.0 * np.pi * f / 299792458.0)**2
        self.Sc_over_Sn = np.asarray(np.maximum(Sc, 0.0)) / cp.asnumpy(self.psd_g)

    def one(self, rng):
        """One (window, params) sample. Returns feats (W,4) float32, meta dict."""
        f0 = rng.uniform(UCB_F_LO_MHZ, UCB_F_HI_MHZ) * 1e-3
        k0 = f0 / self.df
        det_err = rng.uniform(-DET_ERR_BINS, DET_ERR_BINS)
        c_bin = int(round(k0 + det_err))                 # window center (detected)
        s0 = c_bin - W // 2
        if s0 < 1 or s0 + W >= self.n_bins:
            return None
        p = np.array([rng.uniform(*UCB_LOG10A), f0 * 1e3, FDOT_LOG10,
                      rng.uniform(0, 2*np.pi), rng.uniform(-1, 1),
                      rng.uniform(0, np.pi), rng.uniform(0, 2*np.pi),
                      rng.uniform(-1, 1)])
        try:
            m = ucb(f_lims=np.array([(s0) * self.df, (s0 + W) * self.df]),
                    f_inds=[s0, s0 + W], gpu=True, tobs=self.t_obs, dt=self.dt)
            a_r, e_r, _ = m.eval(p.reshape(1, 8))
        except Exception:
            return None
        # GBGPU centers the returned band on the SOURCE (start_inds), ignoring
        # our window start: paste at the true offset so the source lands at its
        # physical in-window position (k0 - s0) — this carries the df signal.
        si = int(cp.asnumpy(cp.asarray(m._template.start_inds)).ravel()[0])
        a_r = cp.asarray(a_r).ravel(); e_r = cp.asarray(e_r).ravel()
        L = a_r.shape[0]
        a = cp.zeros(W, dtype=cp.complex128); e = cp.zeros(W, dtype=cp.complex128)
        lo = max(s0, si); hi = min(s0 + W, si + L)
        if hi - lo < L // 2:                       # source mostly outside window
            return None
        a[lo - s0:hi - s0] = a_r[lo - si:hi - si]
        e[lo - s0:hi - s0] = e_r[lo - si:hi - si]
        # whiten by instrument PSD in-window
        w = self.sqrt_psd_g[s0:s0 + W]
        aw = a / w; ew = e / w
        snr = float(cp.sqrt(4 * self.df * cp.sum(cp.abs(aw)**2 + cp.abs(ew)**2)))
        feats = cp.stack([aw.real, aw.imag, ew.real, ew.imag], -1).astype(cp.float32)
        meta = dict(log10A=float(p[0]), f0=f0, df_bins=float(k0 - c_bin),
                    phi0=float(p[3]), cos_iota=float(p[4]), psi=float(p[5]),
                    **{'lambda': float(p[6])}, sin_beta=float(p[7]),
                    snr=snr, c_bin=c_bin,
                    conf_ratio=float(self.Sc_over_Sn[c_bin]))
        return cp.asnumpy(feats), meta

    def generate(self, n_train, n_val, n_test, seed=SEED):
        rng = np.random.RandomState(seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        info = {'df': self.df, 't_obs': self.t_obs, 'W': W, 'sigma0': self.sigma0,
                'f_mhz': [UCB_F_LO_MHZ, UCB_F_HI_MHZ], 'log10A': list(UCB_LOG10A),
                'fdot_log10': FDOT_LOG10, 'det_err_bins': DET_ERR_BINS,
                'param_order': PARAM_ORDER}
        (self.output_dir / 'dataset_info.json').write_text(json.dumps(info, indent=2))
        for split, n in [('train', n_train), ('val', n_val), ('test', n_test)]:
            if n == 0:
                continue
            sd = self.output_dir / split
            sd.mkdir(parents=True, exist_ok=True)
            feats_all = np.empty((n, W, 4), dtype=np.float32)
            metas = []
            pbar = tqdm(total=n, desc=split)
            got = 0
            while got < n:
                out = self.one(rng)
                if out is None:
                    continue
                feats_all[got] = out[0]
                out[1]['idx'] = got
                metas.append(out[1])
                got += 1
                pbar.update(1)
            pbar.close()
            np.save(sd / 'feats.npy', feats_all)       # single array: fast load
            (sd / 'metadata.json').write_text(json.dumps(metas))
            snrs = [m['snr'] for m in metas]
            print(f"  {split}: {got}, SNR [{min(snrs):.1f}, {max(snrs):.1f}] "
                  f"median {np.median(snrs):.1f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', default='data/ucb')
    ap.add_argument('--n-train', type=int, default=1000000)
    ap.add_argument('--n-val', type=int, default=20000)
    ap.add_argument('--n-test', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    gen = UCBPEGenerator(output_dir=args.output_dir)
    gen.generate(args.n_train, args.n_val, args.n_test, seed=args.seed)
