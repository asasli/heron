# heron/: orchestration

| file | role |
|---|---|
| `cli.py` | the `heron` command: `download`, `check`, `infer`, `train`, `validate`, `figures`, `info` |
| `api.py` | Python interface: `Heron.from_pretrained(...).run(...)` |
| `download.py` | fetches the released archives, verifies their checksums and unpacks them |
| `pipeline.py` | the four stages on one simulated event: `python -m heron.pipeline --simulate --case 1 --n-ucb 30 --out runs/case1` |
| `stages.py` | the stage functions used by the pipeline: `simulate_event`, `detect_oracle`, `detect_yolo`, `alert_posterior`, `recalibrate_alert`, `subtract_catalogue`, `handoff_files`, `run_stages_1_to_3` |
| `run.py` | data generation, training, calibration, export and validation from one YAML file: `python -m heron.run --config config/heron.yaml --stage all` |
| `artifacts.py` | locations and checksums of the weights, data, products and outputs; `python -m heron.artifacts` verifies the weights |
| `check.py` | installation check: `python -m heron.check` |
| `recal.py` | post-hoc calibration map (numpy only) |
| `skyarea.py` | the canonical sky-area estimator of the pointing tier |
| `sources.py` | the three Sangria binaries used in the paper |

## Pipeline outputs

`python -m heron.pipeline --simulate --out <dir>` writes

- `summary.json`: SNR, detection counts, alert sky area (raw and calibrated),
  5/50/95% quantiles of the eleven parameters with the injected values, number
  of subtracted UCBs, residual amplitude, wall time of each stage;
- `alert_posterior.npz`: `samples` (calibrated), `samples_raw`, `param_names`,
  `truth`, `sky90_deg2`;
- `ucb_catalogue.json`: per subtracted source the fitted frequency, the flow
  frequency, sky position, fitted SNR and window residual, with the injected
  list and the detection record;
- `<tag>_{clean,foreground,subtracted}.npz` and `<tag>_README.txt`: the stage-4
  hand-off (format: `pointing/README.md`);
- `pointing/`: stage-4 products when the HyperWave interpreter is available
  (`docs/INSTALL.md`).

Options: `--case {1,2,3}`, `--n-ucb`, `--seed` (noise realization), `--t-pre`
(hours between the end of the data and the merger), `--detector {oracle,yolo}`,
`--ref-scale`, `--skip-pointing`, `--seed-mode {alert,none}`.

## Driver stages

`python -m heron.run --stage <list>` accepts `yolo-data`, `yolo-train`,
`ucb-data`, `ucb-train`, `ucb-response`, `mbhb-data`, `mbhb-train`, `recal`,
`export`, `infer`, `validate`, and the groups `data`, `train`, `all`. Each
training stage copies its artifact into `<out>/weights` under the released file
name; `export` adds `SHA256SUMS` and `manifest.json`. `--dry-run` prints the
commands. `--weights released` makes `infer` and `validate` use the released
weights instead of the bundle of the run.
