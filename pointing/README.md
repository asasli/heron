# pointing/: stage 4, exact inference on the subtracted data

| file | role | environment |
|---|---|---|
| `export.py` | stages 1 to 3 for one simulated event and the hand-off files | `heron` |
| `run_ensemble.sh` | hand-off files for several noise realizations, then stage 4 on each | both |
| `synergy_pe.py` | alert-seeded parallel-tempered MCMC on a hand-off file | `heron-pointing` |
| `verify_conventions.py` | waveform-space checks between the two codes and the frame-mapped injection | `heron-pointing` |
| `aggregate_ensemble.py` | statistics over an ensemble of stage-4 results | either |

The stage-4 scripts import the public HyperWave package
(`hyperwave.detectors.lisa.smbhb`: the MBHB template in its sampling
convention, the priors, the canonical sky-area estimator); see
`docs/INSTALL.md`, "Stage 4".

```bash
# heron environment: the hand-off of the Case-1 event (identical to paper_products/handoff/)
python pointing/export.py --case 1 --n-ucb 30 --tag sangria_synergy --out exports

# heron-pointing environment: stage 4, seeded and unseeded
python pointing/synergy_pe.py --npz exports/sangria_synergy_subtracted.npz --force-backend cuda12x
python pointing/synergy_pe.py --npz exports/sangria_synergy_subtracted.npz --force-backend cuda12x --seed-mode none
python pointing/verify_conventions.py --npz exports/sangria_synergy_subtracted.npz \
    --samples results/pointing/synergy_subtracted_alert.npz --map-truth
```

## The sampler

`synergy_pe.py`: 40 walkers, 10 temperatures (Eryn), stretch moves, and for the
seeded run a Gaussian-mixture proposal for (longitude, sin latitude) fitted to
the alert posterior (25% of the moves). The proposal density enters the
Metropolis-Hastings ratio in both directions and the chains are only
initialized from alert draws, so the target is the posterior under the
likelihood of the hand-off file: a Gaussian (Whittle) likelihood with the
recommended PSD. The sky area is recorded at checkpoints and the run stops when
it is stable over three of them. `--smoke` runs a few iterations as a
functional test.

These are the runs behind the single-realization areas of the paper (191, 134
and 143 deg² at 90% for the clean, foreground and subtracted data; 772 deg²
unseeded at matched wall time; chains in `paper_products/pointing/`). The
ensemble with the hyperbolic and the noise-fit Gaussian likelihood (Fig. 5c)
was run with scripts of the HyperWave repository; its chains are distributed
separately (`docs/DATA.md`) and `evaluation/fig_pointing_ensemble.py` recomputes
the areas from them.

## Hand-off format

Three files per event (band 0.1 to 4 mHz), `<tag>_{clean,foreground,subtracted}.npz`:

```
freqs [Hz], df, Tobs, dt        grid
dA, dE [complex]                raw (unwhitened) TDI A and E data
psdA, psdE                      recommended one-sided PSD for that file
psd_instr, psd_conf             instrument and confusion PSD (S_total = S_instr + S_conf)
hA_inj, hE_inj                  the noiseless injected MBHB template
injection, conventions          injected parameters; time axis, orbits, frame, TDI, waveform
alert_samples [N, 11]           stage-2 alert posterior, columns alert_param_names:
                                Mc, q, chi1, chi2, logD, cos_iota, sin_beta, lambda, psi, phi0, t_pre_hours
track_t, track_f_{lo,med,hi}    Newtonian chirp track, 5/50/95% quantiles over the alert
n_ucb_injected, n_ucb_subtracted, global_alpha
```

| file | content | PSD |
|---|---|---|
| `<tag>_clean.npz` | MBHB + instrument noise | instrument |
| `<tag>_foreground.npz` | + resolved UCBs + confusion realization | instrument + confusion |
| `<tag>_subtracted.npz` | the foreground data after removal of the resolved UCBs | instrument + confusion |

The alert posterior was computed from the same data. It may initialize chains
and serve as a Metropolis-Hastings proposal; it must not be used as a prior.

## Conventions between the two codes

The frame of the stage-4 template differs from the frame of stages 1 to 3 by
more than an epoch shift, because the two BBHx versions use different orbit and
response conventions. Parameters are therefore never mapped analytically:
`verify_conventions.py --map-truth` finds the injection in the stage-4 frame by
maximizing the match with `hA_inj`, `hE_inj` (Case 1: longitude 5.0639 rad,
sin latitude -0.2927, match 0.99997). Sky areas are quoted with the canonical
estimator (`heron/skyarea.py`).
