"""
Resolved galactic-binary population after one month, from the full Sangria
catalogue (Methods, "Realistic foreground density at one month").

All 2.9e7 binaries of the LISA Data Challenge 2a "Sangria" training set
(detached + interacting + verification) are pre-filtered by amplitude
(A > 1e-23, a generous margin below resolvability) and the survivors receive
an exact one-month signal-to-noise ratio with the GPU waveform against the
instrument + confusion PSD. Prints the resolved counts per threshold and the
occupancy per 110-uHz detection tile, and writes sangria_1month_snr.npz
(snr, f0, amp), the population used by evaluation/fig_fullband_scan.py.

The catalogue is the public file LDC2_sangria_training_v2.h5 (3 GB) from the
LISA Data Challenge portal, https://lisa-ldc.lal.in2p3.fr (not redistributed
here); the resulting product is in paper_products/.

    python evaluation/resolved_population_1month.py --catalog /path/to/LDC2_sangria_training_v2.h5
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import h5py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from heron.artifacts import product_out
from gbgpu.gbgpu import GBGPU
from config import galactic_confusion_psd
from data_utils import noise_model
from yolo.yolo_config import SA, SI

YRSID = 3.15576e7
Tobs, dt = YRSID / 12, 15.0
df = 1.0 / Tobs
N = 128
F_LO, F_HI = 1e-4, 6e-3

_ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
_ap.add_argument('--catalog', required=True, help='LDC2_sangria_training_v2.h5')
_args = _ap.parse_args()
f5 = h5py.File(_args.catalog, 'r')
cols = ['Amplitude', 'Frequency', 'FrequencyDerivative', 'Inclination',
        'InitialPhase', 'Polarization', 'EclipticLongitude', 'EclipticLatitude']
parts = []
for grp in ('dgb', 'igb', 'vgb'):
    cat = f5[f'sky/{grp}/cat'][:].squeeze()
    A, F = cat['Amplitude'].astype(float), cat['Frequency'].astype(float)
    keep = (A > 1e-23) & (F > F_LO) & (F < F_HI)
    parts.append({c: cat[c].astype(float)[keep] for c in cols})
    print(f"{grp}: {len(cat)} total, {keep.sum()} pass prefilter (A>1e-23, in band)")
cat = {c: np.concatenate([p[c] for p in parts]) for c in cols}
n = len(cat['Amplitude'])
print(f"prefiltered candidates: {n}")

nbins = int(6.5e-3 / df) + N + 8
grid = np.arange(nbins) * df
nm = noise_model(grid)
psd = nm.eval([SA, SI])
psd = np.asarray(psd.get() if hasattr(psd, 'get') else psd).ravel()
psd_tot = psd + galactic_confusion_psd(grid)
psd_tot[psd_tot <= 0] = np.inf; psd_tot[0] = np.inf

gb = GBGPU(use_gpu=True)
import cupy as cp
psd_g = cp.asarray(psd_tot)
snr = np.zeros(n)
B = 20000
for i in range(0, n, B):
    s = slice(i, min(i + B, n))
    gb.run_wave(cat['Amplitude'][s], cat['Frequency'][s],
                cat['FrequencyDerivative'][s], np.zeros(s.stop - s.start),
                cat['InitialPhase'][s], cat['Inclination'][s],
                cat['Polarization'][s], cat['EclipticLongitude'][s],
                cat['EclipticLatitude'][s], N=N, dt=dt, T=Tobs)
    A, E = cp.asarray(gb.A), cp.asarray(gb.E)
    si = cp.asarray(gb.start_inds).reshape(-1, 1)
    idx = cp.clip(si + cp.arange(N)[None, :], 0, nbins - 1)
    s2 = 4 * df * cp.sum((cp.abs(A) ** 2 + cp.abs(E) ** 2) / psd_g[idx], axis=1)
    snr[s] = cp.asnumpy(cp.sqrt(s2))

f0s = cat['Frequency']
OUT = product_out('sangria_1month_snr.npz')
np.savez(OUT, snr=snr, f0=f0s, amp=cat['Amplitude'])
print(f'saved {OUT}')
for thr in (7, 10, 15):
    m = snr > thr
    inb = m & (f0s > 1e-4) & (f0s < 4e-3)
    print(f"\n1-month SNR>{thr}: {m.sum()} resolved in 0.1-6 mHz; "
          f"{inb.sum()} in the SMBHB band 0.1-4 mHz")
    edges = np.arange(1e-4, 6e-3, 110e-6)
    h, _ = np.histogram(f0s[m], bins=edges)
    nz = h[h > 0]
    if len(nz):
        print(f"  per 110-uHz tile: max {h.max()}, p90(occupied) "
              f"{np.percentile(nz, 90):.0f}, median(occupied) {np.median(nz):.0f}; "
              f"tiles >13: {(h > 13).sum()}, tiles >40: {(h > 40).sum()}")
        top = np.argsort(h)[::-1][:6]
        print("  busiest (mHz:count):",
              {f"{edges[t]*1e3:.2f}": int(h[t]) for t in top})
