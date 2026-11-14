"""
Calibrate the whitened+binned magnitude of a resolved UCB vs frequency, so the
training loop can add realistic UCB contamination ON THE FLY (resampled every
batch) instead of baking it in. A UCB at f0 with amplitude 10^a contributes a
complex value ~ 10^a * g(bin(f0)) to one output bin; we measure g per bin.

Writes results/products/ucb_response.npz: g_bin (per output bin, whitened
binned magnitude per unit amplitude), bin_centers, and the instrument and
confusion fractions per bin. The released file is weights/ucb_response.npz;
copy a regenerated one there before retraining the alert flow.

Run: python evaluation/calibrate_ucb.py
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from heron.artifacts import (MBHB_FLOW, MBHB_NORM, MBHB_RECAL, UCB_FLOW, UCB_RESPONSE,
                             MBHB_DATA, UCB_DATA, load_kernel, product_out,
                             product_in, figure_out)
import cupy as cp
from pe_smbhb.data_mbhb import HetDataGenerator, N_BINS_OUT
from pe_smbhb.data_mbhb_ucb import (UCB_F_LO_MHZ, UCB_F_HI_MHZ,
                                              UCB_HALFWIDTH_HZ)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import NGB_PATH
sys.path.insert(0, NGB_PATH)
from data_utils import ucb

gen = HetDataGenerator(output_dir=None)
A_REF = -21.0  # reference log10A for the linear response

# sweep frequencies across the UCB band; measure peak binned whitened |val|
f_mhz = np.linspace(UCB_F_LO_MHZ, UCB_F_HI_MHZ, 120)
out_bin = np.zeros(len(f_mhz), int)
g_A = np.zeros(len(f_mhz))
g_E = np.zeros(len(f_mhz))
np.random.seed(0)
for i, fm in enumerate(f_mhz):
    f0 = fm * 1e-3
    f_lims = np.array([f0 - UCB_HALFWIDTH_HZ, f0 + UCB_HALFWIDTH_HZ])
    s0 = int(f_lims[0] / gen.df); s1 = int(f_lims[1] / gen.df)
    if s0 < 1 or s1 >= gen.n_bins:
        continue
    # average response magnitude over random extrinsics (antenna pattern)
    magsA, magsE = [], []
    for _ in range(8):
        m = ucb(f_lims=f_lims, f_inds=[s0, s1 + 1], gpu=True,
                tobs=gen.t_obs, dt=gen.dt)
        p = np.array([A_REF, fm, -16.5, np.random.uniform(0, 2*np.pi),
                      np.random.uniform(-1, 1), np.random.uniform(0, np.pi),
                      np.random.uniform(0, 2*np.pi), np.random.uniform(-1, 1)])
        a, e, _ = m.eval(p.reshape(1, 8))
        a = cp.asarray(a).ravel(); e = cp.asarray(e).ravel()
        uA = cp.zeros(gen.n_bins, cp.complex128); uE = cp.zeros(gen.n_bins, cp.complex128)
        uA[s0:s0+len(a)] = a; uE[s0:s0+len(e)] = e
        wA = (uA / gen.sqrt_psd_g)[None, :]
        wE = (uE / gen.sqrt_psd_g)[None, :]
        bA = gen.bin_mean(wA)[0]; bE = gen.bin_mean(wE)[0]
        kA = int(cp.argmax(cp.abs(bA))); out_bin[i] = kA
        magsA.append(float(cp.abs(bA[kA]))); magsE.append(float(cp.abs(bE[kA])))
    g_A[i] = np.median(magsA) / (10**A_REF)
    g_E[i] = np.median(magsE) / (10**A_REF)

# map to per-output-bin response (interpolate onto all bins in the UCB band)
bin_centers = np.array([gen.freqs[a:b].mean()
                        for a, b in zip(gen.edges[:-1], gen.edges[1:])]) * 1e3  # mHz
valid = g_A > 0
g_bin = np.zeros(N_BINS_OUT)
in_band = (bin_centers >= UCB_F_LO_MHZ) & (bin_centers <= UCB_F_HI_MHZ)
g_bin[in_band] = np.interp(bin_centers[in_band], f_mhz[valid], g_A[valid])
# instrument vs confusion fraction per output bin (both whitened by total PSD)
# so training can resample confusion at a VARYING amplitude on the fly while
# keeping the instrument level fixed: noise = instr + conf_amp * conf.
sys.path.insert(0, NGB_PATH)
from data_utils import noise_model
from config import SA, SI
_pi = noise_model(gen.freqs).eval([SA, SI])
psd_instr = np.asarray(_pi.get() if hasattr(_pi, 'get') else _pi).ravel()
f = gen.freqs
psd_conf = 9e-45 * f**(-7/3) * np.exp(-f**0.138 - 221 * f * np.sin(521 * f))
psd_conf *= (1 + np.tanh(1680 * (0.00113 - f))); psd_conf *= (2*np.pi*f/299792458.0)**2
psd_conf = np.maximum(psd_conf, 0.0)
psd_tot = psd_instr + psd_conf
ri = psd_instr / psd_tot
rc = psd_conf / psd_tot
instr_frac = np.zeros(N_BINS_OUT); conf_frac = np.zeros(N_BINS_OUT)
for k, (a, b) in enumerate(zip(gen.edges[:-1], gen.edges[1:])):
    instr_frac[k] = np.sqrt(np.mean(ri[a:b]))
    conf_frac[k] = np.sqrt(np.mean(rc[a:b]))
OUT = product_out('ucb_response.npz')
np.savez(OUT, g_bin=g_bin, bin_centers=bin_centers,
         f_mhz=f_mhz, g_A=g_A, g_E=g_E, A_ref=A_REF,
         instr_frac=instr_frac.astype(np.float32),
         conf_frac=conf_frac.astype(np.float32))
print(f"  confusion fraction: median {np.median(conf_frac):.2f}, "
      f"max {conf_frac.max():.2f} (1.0 = confusion-dominated bin)")
print(f"calibrated UCB response: g_A median {np.median(g_A[valid]):.3e} "
      f"(per 10^log10A), {in_band.sum()} output bins in UCB band "
      f"[{UCB_F_LO_MHZ},{UCB_F_HI_MHZ}] mHz")
print(f"  at log10A=-21 a UCB binned mag ~ {np.median(g_A[valid])*1e-21:.3e} "
      f"(vs per-bin noise std ~ {gen.sigma0/np.sqrt(gen.counts.mean()):.1f})")
print(f"saved {OUT}")
