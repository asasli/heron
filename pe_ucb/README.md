# pe_ucb/: stage 3, the UCB catalogue

A small flow (785,128 parameters) returns the posterior of one galactic binary
from a 128-bin window of whitened A and E data around a detection:
amplitude, frequency offset, phase, inclination, polarization and sky position.
The frequency derivative is fixed at the population value, since it is not
measurable in one month.

| file | role |
|---|---|
| `data_ucb.py` | training windows: `python -m pe_ucb.data_ucb --output-dir data/ucb --n-train 1000000 --seed 42` |
| `train_flow.py` | model (`UCBFlow`), augmentation (instrument noise, confusion level ladder, neighbouring lines) and training: `python -m pe_ucb.train_flow --data data/ucb --epochs 40` |

The window centre is the detection frequency with an error of up to 8 bins
(3 µHz), the measured error of the detector; the flow infers the offset.

Subtraction (`heron/stages.py::fstat_subtract`). The UCB waveform is linear in
four amplitude combinations. For each source the frequency is scanned in two
stages around the flow estimate (13 points over ±1.5 bins, then 13 points over
±0.15 bins) and, per TDI channel, the data are projected on the rank-4 singular
basis of a set of waveforms generated at that frequency; the best projection is
subtracted. The extrinsic parameters are nearly degenerate over one month and
the subtraction does not use them.

Validations: `evaluation/eval_ucb_flow.py` (calibration, frequency precision,
throughput), `evaluation/ucb_subtract_test.py --fstat` (residual after
subtraction), `evaluation/fig_ucb_recovery.py` (Extended Data Fig. 5).
