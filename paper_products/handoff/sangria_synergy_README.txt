HERON -> HyperWave export. Raw d(f) + one-sided PSD, band 0.1-4 mHz, T_obs=YRSID/12, A/E TDI. Whittle lnL: -2 df Sum |dA-hA|^2/psdA + (A->E).
clean = MBHB+noise (ideal); foreground = +UCBs+confusion (real); subtracted = foreground after HERON removal of RESOLVED UCBs.
psdA/psdE = the RECOMMENDED PSD per file: instrument-only for clean; instrument+confusion (psd_total) for foreground/subtracted, because the unresolved confusion is stationary noise and belongs in the PSD (resolved lines cannot be treated as noise and ARE subtracted). Keys psd_instr and psd_conf are provided separately; feel free to FIT the noise level per your Gaussian-likelihood noise-fitting analysis.

alert_samples [N,11] (columns alert_param_names) = the HERON stage-2
flow posterior, computed from the SAME data. Safe uses (bias-free):
  1. initialize RJMCMC chains from these draws (cuts burn-in);
  2. MH independence proposals for (lambda, beta) and wavelet birth
     moves along the chirp track track_f_med(track_t) [5-95% band
     track_f_lo..track_f_hi] -- always with the MH correction.
Do NOT multiply it in as a prior: it would double-count the data.

CONVENTIONS (also in the `conventions` dict inside each npz):
  t=0 of dA/dE = t=0 of the BBHx analytic equal-armlength orbit model
  (zero initial constellation longitude/rotation phase); window
  [0, Tobs], merger at injection['t_ref_s'] seconds. (lam, sinb) are
  SSB-ecliptic at that epoch. TDI 1st generation AET. Waveform BBHx
  PhenomD (2,2); f_ref=0 -> fmax internally; phi0/t_ref defined at
  f_ref. The full injected extrinsics (iota, psi, phi0) are in
  injection. hA_inj/hE_inj = the NOISELESS injected template on the
  same grid: verify truth-in-credible-region against it directly,
  or calibrate your orbit-epoch mapping by matched-filtering it in
  your own response convention.
