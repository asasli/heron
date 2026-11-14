"""Stage 4 — alert-seeded SMBHB PE on a HERON hand-off npz (HyperWave engine).

Runs the 11-parameter PE on an exported data set (clean / foreground /
subtracted) with the per-file PSD, and times how quickly the sky localizes.
Seeding follows the bias-free hand-off recipe (see docs/INSTALL.md, "Stage 4"):

  * chains INITIALIZED from the stage-2 alert posterior draws (cuts burn-in;
    never changes the stationary distribution);
  * a GMM fit to the alert (lambda, sin beta) used as an MH INDEPENDENCE
    proposal mixed with stretch moves -- the proposal density enters the MH
    ratio in both directions, so the target stays exactly the posterior.

The alert posterior is NOT multiplied into the target (it came from the same
data). --seed-mode none gives the unseeded control for the wall-clock A/B.

Requires the public HyperWave package (NOT in the base HERON environment —
see docs/INSTALL.md, "Stage 4"; a pip bbhx build may conflict with a source-built
one, so prefer a separate environment or the cluster environment):
    pip install "git+https://github.com/asasli/HyperWave.git" eryn scikit-learn

Usage:
    python pointing/synergy_pe.py --npz exports/<tag>_subtracted.npz \
        [--force-backend cuda12x] [--seed-mode none] [--smoke]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    from hyperwave.detectors.lisa.smbhb import (
        SMBHBbbhxTemplate, sampling_to_physical, prior_bounds,
        SAMPLING_NAMES, PERIODIC_IDX, TRUTH_PHYS, sky_credible_area,
        YRSID_SI as YRSID)
except ImportError:
    sys.exit("The HyperWave package is required for stage 4:\n"
             '  pip install "git+https://github.com/asasli/HyperWave.git" '
             "eryn scikit-learn\n(prefer a separate env; see docs/INSTALL.md, Stage 4)")


def cred_area_fast(lon, sinlat, p, nbins=(360, 180)):
    """Cheap checkpoint-curve estimator (raw histogram; reads ~30% low —
    final numbers below use the canonical sky_credible_area)."""
    H, xe, ye = np.histogram2d(lon % (2 * np.pi), np.clip(sinlat, -1, 1),
                               bins=list(nbins), range=[[0, 2 * np.pi], [-1, 1]])
    h = np.sort(H.ravel())[::-1]
    c = np.cumsum(h) / h.sum()
    nsel = int(np.searchsorted(c, p) + 1)
    return nsel * (xe[1] - xe[0]) * (ye[1] - ye[0]) * (180 / np.pi) ** 2


def alert_to_sampling(a, tobs_h):
    """Map HERON alert_samples columns -> the PE sampling order."""
    out = np.empty((a.shape[0], 11))
    out[:, 0:5] = a[:, 0:5]                    # Mc, q, chi1, chi2, log10D
    out[:, 5] = tobs_h - a[:, 10]              # tc_h = Tobs_h - t_pre_hours
    out[:, 6] = a[:, 5]                        # cos_iota
    out[:, 7] = a[:, 6]                        # sin_beta
    out[:, 8] = a[:, 7]                        # lambda
    out[:, 9] = a[:, 8]                        # psi
    out[:, 10] = a[:, 9]                       # phi0
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", required=True, help="HERON hand-off npz")
    p.add_argument("--seed-mode", choices=["alert", "none"], default="alert")
    p.add_argument("--gmm-weight", type=float, default=0.25)
    p.add_argument("--force-backend", default=None)
    p.add_argument("--nwalkers", type=int, default=40)
    p.add_argument("--ntemps", type=int, default=10)
    p.add_argument("--seg", type=int, default=250, help="iterations per checkpoint")
    p.add_argument("--max-steps", type=int, default=8000)
    p.add_argument("--thin", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="tiny run (setup test)")
    p.add_argument("--out-dir", default="results/pointing")
    p.add_argument("--outfile", default=None)
    args = p.parse_args()
    if args.smoke:
        args.nwalkers, args.ntemps, args.seg, args.max_steps, args.thin = 26, 4, 30, 60, 1

    rng = np.random.default_rng(args.seed)
    z = np.load(args.npz, allow_pickle=True)
    freqs = np.asarray(z["freqs"]); df = float(z["df"])
    dA, dE = np.asarray(z["dA"]), np.asarray(z["dE"])
    psdA, psdE = np.asarray(z["psdA"]), np.asarray(z["psdE"])
    tobs = float(z["Tobs"]); tobs_h = tobs / 3600.0
    inj = z["injection"].item()
    tag = Path(args.npz).stem.split("_")[-1]
    print(f"[setup] {tag}: {freqs.size} bins "
          f"[{freqs.min()*1e3:.2f},{freqs.max()*1e3:.2f}] mHz  "
          f"Tobs={tobs/86400:.2f} d  SNR={float(z['mbhb_snr']):.1f}", flush=True)

    template = SMBHBbbhxTemplate(freqs, t_obs_years=tobs / YRSID,
                                 force_backend=args.force_backend)

    # sanity: template at the injection must reproduce ~the quoted SNR
    truth = np.array([inj["Mc"], inj["q"], inj["chi1"], inj["chi2"], inj["logD"],
                      tobs_h - inj["t_pre_hr"], np.cos(TRUTH_PHYS["inc"]),
                      inj["sinb"], inj["lam"], TRUTH_PHYS["psi"],
                      TRUTH_PHYS["phi_ref"]])
    for k, j in (("cos_iota", 6), ("cosi", 6), ("psi", 9), ("phi0", 10)):
        if k in inj:
            truth[j] = inj[k]
    h = template.batch_eval(sampling_to_physical(truth[None]))[0, :2, :]
    snr = np.sqrt(4 * df * np.sum(np.abs(h[0]) ** 2 / psdA
                                  + np.abs(h[1]) ** 2 / psdE))
    print(f"[sanity] template SNR at injection = {snr:.1f} (file says "
          f"{float(z['mbhb_snr']):.1f})", flush=True)

    inv_pA, inv_pE = 1.0 / psdA, 1.0 / psdE

    def log_like(x):
        x = np.atleast_2d(x)
        h = template.batch_eval(sampling_to_physical(x))[:, :2, :]
        rA = dA[None] - h[:, 0]; rE = dE[None] - h[:, 1]
        return (-2.0 * df * (np.sum(np.abs(rA) ** 2 * inv_pA[None], axis=1)
                             + np.sum(np.abs(rE) ** 2 * inv_pE[None], axis=1)))

    lo, hi = prior_bounds()
    ndim = len(SAMPLING_NAMES)

    from eryn.prior import ProbDistContainer, uniform_dist
    from eryn.ensemble import EnsembleSampler
    from eryn.state import State
    from eryn.moves import StretchMove, MHMove

    priors = {"model_0": ProbDistContainer(
        {i: uniform_dist(lo[i], hi[i]) for i in range(ndim)})}

    moves = [(StretchMove(), 1.0)]
    if args.seed_mode == "alert":
        if "alert_samples" not in z:
            sys.exit("npz has no alert_samples — re-export with pointing/export.py "
                     "(or run with --seed-mode none)")
        from sklearn.mixture import GaussianMixture
        alert = alert_to_sampling(np.asarray(z["alert_samples"], float), tobs_h)
        gmm = GaussianMixture(8, random_state=1).fit(alert[:, [8, 7]])  # (lam, sinb)

        class GMMSkyMove(MHMove):
            """Independence proposal for (lambda, sin beta) from the alert GMM.

            factors = log q(cur) - log q(new): the proposal density enters the
            MH ratio both ways, so the stationary distribution is untouched.
            Out-of-prior proposals are rejected by the prior (no clipping)."""
            def get_proposal(self, branches_coords, random, *a_, **k_):
                q = {}
                factors = None
                for name, c in branches_coords.items():
                    nt, nw = c.shape[:2]
                    newc = c.copy()
                    s, _ = gmm.sample(nt * nw)
                    cur = np.stack([c[:, :, 0, 8].ravel(),
                                    c[:, :, 0, 7].ravel()], axis=1)
                    lqc = gmm.score_samples(cur)
                    lqn = gmm.score_samples(s)
                    newc[:, :, 0, 8] = s[:, 0].reshape(nt, nw)
                    newc[:, :, 0, 7] = s[:, 1].reshape(nt, nw)
                    q[name] = newc
                    factors = (lqc - lqn).reshape(nt, nw)
                return q, factors

        moves = [(StretchMove(), 1.0 - args.gmm_weight),
                 (GMMSkyMove(), args.gmm_weight)]
        i0 = rng.integers(len(alert), size=args.ntemps * args.nwalkers)
        coords = alert[i0].reshape(args.ntemps, args.nwalkers, ndim)
        coords = np.clip(coords, lo[None, None] + 1e-9, hi[None, None] - 1e-9)
    else:
        coords = np.column_stack([rng.uniform(lo[i], hi[i],
                                  args.ntemps * args.nwalkers)
                                  for i in range(ndim)]
                                 ).reshape(args.ntemps, args.nwalkers, ndim)

    sampler = EnsembleSampler(
        args.nwalkers, ndim, log_like, priors,
        tempering_kwargs=dict(ntemps=args.ntemps), vectorize=True,
        periodic={"model_0": PERIODIC_IDX}, moves=moves)

    state = State(coords[:, :, None, :].reshape(args.ntemps, args.nwalkers, 1, ndim))
    t0 = time.perf_counter()
    times, a90s, a50s = [], [], []
    n_done = 0
    print(f"[run] {tag} seed-mode={args.seed_mode} nw={args.nwalkers} "
          f"nt={args.ntemps} seg={args.seg}", flush=True)
    while n_done < args.max_steps:
        state = sampler.run_mcmc(state, args.seg, burn=0, thin_by=args.thin,
                                 progress=False)
        n_done += args.seg
        ch = sampler.get_chain()["model_0"][:, 0, :, 0, :]      # (ns, nw, ndim)
        half = ch[ch.shape[0] // 2:].reshape(-1, ndim)
        a90 = cred_area_fast(half[:, 8], half[:, 7], 0.9)
        a50 = cred_area_fast(half[:, 8], half[:, 7], 0.5)
        wall = time.perf_counter() - t0
        times.append(wall); a90s.append(a90); a50s.append(a50)
        print(f"  step {n_done:5d}  wall {wall/60:6.1f} min  "
              f"sky90 {a90:9.2f} deg^2  sky50 {a50:8.2f}", flush=True)
        if len(a90s) >= 4:
            recent = a90s[-3:]
            if max(recent) - min(recent) < 0.05 * a90s[-1]:
                print("  [converged: sky area stable over 3 checkpoints]", flush=True)
                break
    wall = time.perf_counter() - t0

    ch = sampler.get_chain()["model_0"][:, 0, :, 0, :]
    half = ch[ch.shape[0] // 2:].reshape(-1, ndim)
    # final numbers at the CANONICAL (paper) estimator: 0.5-deg grid,
    # 1-cell Gaussian smoothing, greedy HPD, both sky modes
    a90c = float(sky_credible_area(half[:, 8], half[:, 7], 0.9))
    a50c = float(sky_credible_area(half[:, 8], half[:, 7], 0.5))
    # time-to-localization: first checkpoint within 10% of the final area
    a_fin = a90s[-1]
    t_loc = next((t for t, a in zip(times, a90s) if abs(a - a_fin) < 0.1 * a_fin),
                 times[-1])
    print(f"\n[{tag} | seed={args.seed_mode}] FINAL (canonical estimator): "
          f"sky90={a90c:.1f} deg^2 sky50={a50c:.1f} | wall={wall/60:.1f} min | "
          f"time-to-localization={t_loc/60:.1f} min", flush=True)
    out = args.outfile or str(Path(args.out_dir) / f"synergy_{tag}_{args.seed_mode}.npz")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, times=np.array(times), a90=np.array(a90s), a50=np.array(a50s),
             a90_canonical=a90c, a50_canonical=a50c,
             samples=half.astype(np.float32), names=SAMPLING_NAMES,
             truth=truth, wall=wall, t_loc=t_loc, tag=tag,
             seed_mode=args.seed_mode)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
