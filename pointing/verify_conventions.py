"""Cross-code conventions check for a HERON hand-off npz (HyperWave engine).

Everything is done in waveform space against the exported noiseless template
hA_inj/hE_inj, so no analytic convention mapping is ever needed:

  (1) sanity          : <d, h_inj> / <h_inj, h_inj>          (expect ~1)
  (2) convention gap  : match( h_ours(injection params), h_inj ) — the pure
      wrapper/version difference (phase+lag maximised);
  (3) epoch calibration: T0 that maximises (2) — the constant offset to apply
      to t_ref in this wrapper for HERON-generated data (known: +92.1 d);
  (4) posterior check : match( h_ours(theta_i), h_inj ) over PE samples —
      median ~1 == waveform-space truth-in-credible-region;
  (5) --map-truth     : optimise the match -> the injected source's parameters
      in THIS wrapper's frame (the Fig.-5 star).

Requires the public HyperWave package (see docs/INSTALL.md, "Stage 4").

Usage:
    python pointing/verify_conventions.py --npz exports/<tag>_subtracted.npz \
        [--samples results/pointing/synergy_subtracted_alert.npz] [--map-truth]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

try:
    from hyperwave.detectors.lisa.smbhb import (
        SMBHBbbhxTemplate, sampling_to_physical, TRUTH_PHYS,
        YRSID_SI as YRSID)
except ImportError:
    sys.exit("The HyperWave package is required:\n"
             '  pip install "git+https://github.com/asasli/HyperWave.git"\n'
             "(prefer a separate env; see docs/INSTALL.md, Stage 4)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", required=True, help="HERON hand-off npz")
    ap.add_argument("--samples", default=None,
                    help="synergy_pe.py output npz (enables checks 4/5)")
    ap.add_argument("--t0-grid", default="0,120,1.0",
                    help="epoch scan lo,hi,step in days")
    ap.add_argument("--map-truth", action="store_true",
                    help="optimize match(h_ours, h_inj) -> mapped truth params")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    if "hA_inj" not in z:
        sys.exit("This npz has no hA_inj/hE_inj — re-export with "
                 "pointing/export.py first.")
    freqs = np.asarray(z["freqs"]); df = float(z["df"]); tobs = float(z["Tobs"])
    dA, dE = np.asarray(z["dA"]), np.asarray(z["dE"])
    pA, pE = np.asarray(z["psdA"]), np.asarray(z["psdE"])
    hAi, hEi = np.asarray(z["hA_inj"]), np.asarray(z["hE_inj"])
    inj = z["injection"].item()
    if "conventions" in z:
        print("[conventions]", z["conventions"].item(), "\n")

    def ip(a, b, w):                       # 4 df Re<a|b> weighted
        return 4 * df * np.real(np.sum(np.conj(a) * b * w))

    hh = ip(hAi, hAi, 1 / pA) + ip(hEi, hEi, 1 / pE)
    dh = ip(dA, hAi, 1 / pA) + ip(dE, hEi, 1 / pE)
    print(f"(1) <d,h_inj>/<h_inj,h_inj> = {dh/hh:.4f}   SNR_inj = {np.sqrt(hh):.1f}")

    tmpl = SMBHBbbhxTemplate(freqs, t_obs_years=tobs / YRSID, force_backend="cpu")
    th = tobs / 3600.0
    base = np.array([inj["Mc"], inj["q"], inj["chi1"], inj["chi2"], inj["logD"],
                     th - inj["t_pre_hr"],
                     inj.get("cos_iota", np.cos(inj.get("iota", TRUTH_PHYS["inc"]))),
                     inj["sinb"], inj["lam"],
                     inj.get("psi", TRUTH_PHYS["psi"]),
                     inj.get("phi0", TRUTH_PHYS["phi_ref"])])

    NPAD = 4 * len(freqs)

    # calibrate the padded-IFFT normalisation on the exact zero-lag identity
    num0 = (np.fft.ifft(np.conj(hAi / pA) * hAi, n=NPAD)
            + np.fft.ifft(np.conj(hEi / pE) * hEi, n=NPAD))
    K = float(np.abs(num0).max()) / hh

    def match_of(x):
        h2 = tmpl.batch_eval(sampling_to_physical(np.atleast_2d(x)))[0, :2, :]
        num = (np.fft.ifft(np.conj(hAi / pA) * h2[0], n=NPAD)
               + np.fft.ifft(np.conj(hEi / pE) * h2[1], n=NPAD))
        n2 = ip(h2[0], h2[0], 1 / pA) + ip(h2[1], h2[1], 1 / pE)
        return float(np.abs(num).max()) / (K * np.sqrt(n2 * hh))

    m0 = match_of(base)
    print(f"(2) match(h_ours(injection), h_inj), no epoch shift = {m0:.4f}")

    lo, hi, st = (float(v) for v in args.t0_grid.split(","))
    best = (0.0, m0)
    for T0 in np.arange(lo, hi + st / 2, st):
        x = base.copy(); x[5] = base[5] + T0 * 24.0
        m = match_of(x)
        if m > best[1]:
            best = (T0, m)
    print(f"(3) epoch calibration: best T0 = {best[0]:.2f} d  ->  match = {best[1]:.4f}")
    for T0 in np.arange(best[0] - 1.0, best[0] + 1.0, 0.05):
        x = base.copy(); x[5] = base[5] + T0 * 24.0
        m = match_of(x)
        if m > best[1]:
            best = (T0, m)
    print(f"    refined: T0 = {best[0]:.3f} d  match = {best[1]:.4f}")

    if args.samples and os.path.exists(args.samples):
        s = np.load(args.samples)["samples"]
        rng = np.random.default_rng(0)
        sub = s[rng.choice(len(s), min(200, len(s)), replace=False)]
        ms = np.array([match_of(x) for x in sub])
        print(f"(4) posterior-vs-h_inj match: median {np.median(ms):.4f} "
              f"90% CI [{np.percentile(ms,5):.4f}, {np.percentile(ms,95):.4f}]")
        print("    (median ~1 => the PE recovered the injected waveform; the sky")
        print("     offset is purely the fixed wrapper convention of (3).)")
        if args.map_truth:
            from scipy.optimize import minimize
            x0 = sub[int(np.argmax(ms))]
            res = minimize(lambda x: -match_of(x), x0, method="Nelder-Mead",
                           options=dict(maxiter=600, xatol=1e-5, fatol=1e-7))
            xb = res.x
            print(f"(5) MAPPED TRUTH (this wrapper frame), match={-res.fun:.5f}:")
            print(f"    lambda* = {xb[8] % (2*np.pi):.4f} rad   sin_beta* = {xb[7]:.4f}")
            np.savez("mapped_truth.npz", mapped_truth=xb, match=-res.fun)
            print("    saved -> mapped_truth.npz")


if __name__ == "__main__":
    main()
