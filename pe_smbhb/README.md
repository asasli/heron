# pe_smbhb/: stage 2, the alert

An amortized posterior over the eleven MBHB parameters (chirp mass, mass ratio,
two aligned spins, distance, inclination, ecliptic latitude and longitude,
polarization, phase, merger time) from one month of A and E data, conditioned
on the number of resolved UCBs.

| file | role |
|---|---|
| `data_mbhb.py` | heterodyned, bin-averaged training data: `python pe_smbhb/data_mbhb.py --output-dir data/mbhb --seed 90000 --wide-spin-ref` |
| `data_mbhb_ucb.py` | generator with resolved UCBs and a confusion realization in the data (used by the pipeline simulation) |
| `networks.py` | parameter names and ranges, encoder building blocks |
| `train_flow.py` | the model (`FlowHet`), the dataset class, the foreground augmentation and the training loop: `python -m pe_smbhb.train_flow --help` |
| `inference.py` | `draw_ref`, `het_features`, `gnpe_marg`: reference draw, heterodyning, pose-iterated sampling |

Representation. The whitened data are multiplied by the conjugate phase of a
reference waveform and averaged into 4096 bins; the input is
(Re r_A, Im r_A, Re r_E, Im r_E, log(1 + |A_ref|)) per bin. Noise is white per
bin in this representation and is added during training, together with a
foreground of up to 40 resolved lines (through the calibrated response
`weights/ucb_response.npz`) and up to 500 faint lines, all resampled every
batch.

Model. A residual 1-D CNN with FiLM conditioning on the reference parameters,
the UCB count and the sky pose, a learned matched-filter bank, and a neural
spline flow with 40 transforms; 108,775,441 parameters. Chirp mass and merger
time are predicted as kernel-scaled residuals to the reference, the other
parameters absolutely.

Inference. The sky pose on which the flow is conditioned is iterated from a
broad seed (`gnpe_marg`, three rounds), with eight pose draws in the last
round. One heterodyne reference is used per event. The post-hoc calibration map
(`heron/recal.py`) is applied to the samples.

Training of the released model: `docs/TRAINING_PROVENANCE.md`. Reproduction of
the calibration figures: `docs/REPRODUCE.md`.
