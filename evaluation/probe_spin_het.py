"""
DECISION PROBE: does the relative-binning representation survive references
with LARGE (prior-wide) spin errors, if (Mc,t) are ridge-compensated?

Why: the model's spin posteriors are reference-anchored (truth-leaking kernel).
The fix — spin-uninformative references — is only possible if a chi-mismatched
(but ridge-compensated) reference still heterodynes the signal into a smooth,
bin-resolvable residual. If YES -> regen data with wide spin kernel + retrain
(v15) gives honest spins. If NO -> the anchoring is a fundamental limitation of
the 4096-bin representation (documented, referee-proof).

Metric: SNR retention of the binned representation,
  rho_ratio = sqrt( sum_k n_k |mean_k(r)|^2 / sum_i |r_i|^2 ),  r = w.h. x c*,
i.e. the fraction of matched-filter SNR that survives bin-averaging (1.0 =
lossless, the phase is slowly varying per bin). Plus median |dphi|/bin.

Run: python evaluation/probe_spin_het.py --n-events 16
"""
import sys, argparse
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from heron.artifacts import (MBHB_FLOW, MBHB_NORM, MBHB_RECAL, UCB_FLOW, UCB_RESPONSE,
                             MBHB_DATA, UCB_DATA, load_kernel, product_out,
                             product_in, figure_out)
import cupy as cp
from pe_smbhb.data_mbhb import HetDataGenerator, FIDUCIAL_EXT, KERNEL

CHI_ERRS = [0.05, 0.15, 0.3, 0.6, 1.0]   # sigma of the spin perturbation


def retention(gen, wA, wE, Aref):
    """SNR fraction retained by bin-averaging the heterodyned channels."""
    mag = cp.abs(Aref)
    c = cp.where(mag > 0, cp.conj(Aref) / cp.where(mag > 0, mag, 1.0), 0.0)
    tot_full, tot_bin = 0.0, 0.0
    drifts = []
    for w in (wA, wE):
        r = w * c
        rb = gen.bin_mean(r[None, :])[0]                     # (4096,)
        tot_full += float(cp.sum(cp.abs(r) ** 2))
        tot_bin += float(cp.sum(gen.counts_g * cp.abs(rb) ** 2))
    # phase drift per output bin (A channel): mean |d arg| within bins ~ between-bin diff
    rbA = gen.bin_mean((wA * c)[None, :])[0]
    live = cp.abs(rbA) > 1e-12
    ph = cp.angle(rbA)
    d = cp.abs(cp.angle(cp.exp(1j * (ph[1:] - ph[:-1]))))
    d = d[live[1:] & live[:-1]]
    drift = float(cp.median(d)) if d.size else float('nan')
    return np.sqrt(tot_bin / max(tot_full, 1e-30)), drift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-events', type=int, default=16)
    args = ap.parse_args()

    gen = HetDataGenerator(output_dir=None)
    np.random.seed(77)
    for _ in range(2):
        Pw, _ = gen.draw_priors(64); gen.gen_wf(Pw)

    # events: decent band, mixed t_pre
    P_all, t_pre_all = [], []
    while len(P_all) < args.n_events:
        P, tp = gen.draw_priors(64)
        for b in range(64):
            if 1.0 < tp[b] < 400 and len(P_all) < args.n_events:
                P_all.append(P[b]); t_pre_all.append(tp[b])
    P_true = np.array(P_all); t_pre = np.array(t_pre_all)
    A, E = gen.gen_wf(P_true)
    wA_all = A / gen.sqrt_psd_g[None, :]
    wE_all = E / gen.sqrt_psd_g[None, :]

    k = KERNEL['base']
    print(f"{'chi_err':>8} {'comp':>5} | {'retention: med':>14} {'min':>6} | {'dphi/bin med [rad]':>18}")
    for chi_err in CHI_ERRS:
        for comp in (True, False):
            rng = np.random.RandomState(int(chi_err * 1000) + int(comp))
            Pp = P_true.copy()
            Pp[:, 1] = np.maximum(Pp[:, 1] * np.exp(rng.randn(len(Pp)) * k['log_q']), 1.005)
            Pp[:, 2] = np.clip(Pp[:, 2] + rng.randn(len(Pp)) * chi_err, -0.99, 0.99)
            Pp[:, 3] = np.clip(Pp[:, 3] + rng.randn(len(Pp)) * chi_err, -0.99, 0.99)
            for col, key in [(4, 'logD'), (6, 'ci'), (7, 'sb'), (8, 'lam'),
                             (9, 'psi'), (10, 'phi0')]:
                Pp[:, col] = FIDUCIAL_EXT[key]
            if comp:
                dmc, dt = gen.ridge_compensate(P_true, Pp, t_pre)
                Pp[:, 0] = P_true[:, 0] + dmc
                Pp[:, 5] = P_true[:, 5] + dt / 3600.0
            Aref, _ = gen.gen_wf(Pp)
            rets, drs = [], []
            for b in range(len(Pp)):
                rr, dd = retention(gen, wA_all[b], wE_all[b], Aref[b])
                rets.append(rr); drs.append(dd)
            rets = np.array(rets)
            print(f"{chi_err:>8.2f} {str(comp):>5} | {np.median(rets):>14.4f} {rets.min():>6.3f} "
                  f"| {np.nanmedian(drs):>18.3f}")

    print("\nVERDICT guide: retention >0.98 at chi_err>=0.6 (comp=True) -> wide-spin-kernel")
    print("regen+retrain is SAFE (honest spin posteriors achievable). retention <0.9 ->")
    print("representation breaks; spin anchoring is a documented structural limitation.")


if __name__ == '__main__':
    main()
