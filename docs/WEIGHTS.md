# Model weights

Archive: `heron_weights_v1.0.tar` (0.5 GB).

```bash
heron download                        # fetch, verify the archive, unpack into weights/, verify every file
heron download --from /path/to/dir    # the same from an archive file already on disk
heron info                            # which weights directory is in use and what it contains
```

`heron infer` and `heron check` call the download themselves when `weights/` is
empty. By hand: unpack the archive in the repository root (`tar -xf
heron_weights_v1.0.tar`) and run `python -m heron.artifacts`, which verifies
every file against `weights/SHA256SUMS`. SHA-256 of the archive:
`80525e0e0a2c4339a59a19de3e56fe197938827b98a8427471a5bd854e07b617`.

Another weights directory, for instance a bundle you trained, is selected with
`heron infer --weights DIR`, with `Heron.from_pretrained(DIR)` or with the
environment variable `HERON_WEIGHTS`.

| file | size | content |
|---|---|---|
| `mbhb_flow.pt` | 483 MB | stage-2 alert flow: model state, architecture configuration, epoch (128), validation losses |
| `mbhb_norm.pt` | 2 kB | per-channel mean and standard deviation of the input features |
| `mbhb_dataset_info.json` | 1 kB | frequency grid, binning, reference kernel and parameter order of the training set |
| `recal_map.npz` | 0.3 MB | post-hoc calibration map: 3000 sorted ranks for each of 12 parameters |
| `ucb_flow.pt` | 3.4 MB | stage-3 UCB window flow (785,128 parameters) with parameter ranges |
| `ucb_response.npz` | 0.1 MB | calibrated per-bin UCB line response and instrument/confusion fractions |
| `yolo_lisa_best.pt` | 5.1 MB | stage-1 detector |
| `yolo26n_base.pt` | 5.3 MB | pretrained base checkpoint the detector was fine-tuned from |

SHA-256:

```
43513808fe762927c63eb20bd307e793bf3139a29b3e82da1333986ab7fe4b95  mbhb_flow.pt
9f8241303180fdcf3c9b41d7f7bf9570d5ff3836c9ebbd0bcb71368019637824  mbhb_norm.pt
ae88835bdd606d6ee85d8b3878f287788b04f02fe0f9792d83ad08e7a939ec0c  mbhb_dataset_info.json
18ab3390281305781b5aef544af0119935cbf46900086a63fcad0b318a5c7975  recal_map.npz
23d0ddf50d155247d6cbeb3f3fc1a7b88320b8942c45c8859d1f9314198d9411  ucb_flow.pt
48b15e74673831996b01039981d2d072f1b3e1d5019407a22c04bf6d6b6ffef2  ucb_response.npz
6957a647efa22e3b9c32d45f635ff23048358a041144b30153adcc48b891414f  yolo_lisa_best.pt
9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef  yolo26n_base.pt
```

The checkpoints are PyTorch pickles written by this code; they are loaded with
`torch.load(..., weights_only=False)`. Load only files whose checksum matches.

A retrained bundle written by `python -m heron.run` has the same file names and
its own `SHA256SUMS` and `manifest.json` (configuration, code commit, package
versions); `export HERON_WEIGHTS=<run>/weights` points the repository at it.
How the released weights were trained: `docs/TRAINING_PROVENANCE.md`.
