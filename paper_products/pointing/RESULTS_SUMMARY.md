# HyperWave <- HERON synergy: PE results (2026-07-13)

Files: synergy_{clean,foreground,subtracted}_alert.npz + synergy_subtracted_none.npz
Keys per file: samples (N,11 cold-chain, HyperWave sampling order Mc,q,chi1,chi2,
log10D,tc_h,cosi,sinb,lam,psi,phi0), times/a90/a50 (checkpoint curves), wall,
t_loc, truth (as mapped), tag, seed_mode.

| dataset     | seed  | sky90 [deg2] | sky50 | wall  | t_loc |
|-------------|-------|--------------|-------|-------|-------|
| clean       | alert | 142.0        | 26.7  | 24.4m | 19.0m |
| foreground  | alert |  98.7        | 22.9  | 20.6m | 15.4m |
| subtracted  | alert |  97.4        | 18.5  | 19.2m | 14.0m |
| subtracted  | none  | 310.7        | 59.8  | 15.1m | (3x broader, unconverged) |

Hardware: 1x A100, bbhx cuda12x, 40 walkers x 10 temps, GMM(8) sky independence
proposal (25%) + stretch, init from alert draws. Bias-free per the hand-off
recipe (proposal density in the MH ratio both ways; alert never used as prior).

CONVENTIONS: fully verified vs the re-export's hA_inj/hE_inj:
posterior-vs-h_inj match = 0.9999/0.9995/0.9998 (clean/fore/sub) -> the PE
recovered the injected waveform on all three (waveform-space truth-in-CR PASS).
Sky coordinates here are in the HYPERWAVE wrapper frame; to map to the HERON
frame apply the epoch calibration t_ref -> t_ref + 92.1 d (match 0.9875), i.e.
the fixed constellation-phase offset between bbhx orbit conventions.

Intrinsics recovery (subtracted, chain best vs injection): Mc +0.14%, chi_eff
preserved (-0.341 -> -0.339), tc +4.5 min, log10 D +0.018.

Reproduce: examples/lisa-mini-global/smbhb_synergy_pe.py (+ slurm
examples/clusters/lisa_sangria_a100.slurm); conventions check:
examples/lisa-mini-global/verify_synergy_conventions.py.
