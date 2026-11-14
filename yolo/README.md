# yolo/: stage 1, detection

A single-stage object detector (YOLO nano, fine-tuned) finds galactic binaries
as horizontal tracks and an MBHB chirp as a vertical feature in the whitened
time-frequency spectrogram.

The detector is trained on one 110 µHz band (2.53 to 2.64 mHz; 30 days at 15 s
sampling, STFT length 16384, 640 x 640 images). Every image is normalized by its
own 1st and 99th percentiles after whitening, so a whitened tile from any band
has the statistics of a training image; at inference the same window slides
across the band with 50% overlap and no retraining
(`inference_fullband.py`). Duplicate detections from overlapping tiles are
merged within 4 µHz. The full-band operating point is a confidence of 0.5, and
the MBHB flag requires that confidence in at least two tiles.

| file | role |
|---|---|
| `yolo_config.py` | band, spectrogram, dataset and training constants |
| `dataset_generator.py` | simulated spectrogram images and labels: `python -m yolo.dataset_generator --output data/yolo` |
| `train_yolo.py` | fine-tuning from `weights/yolo26n_base.pt`: `python -m yolo.train_yolo --data-dir data/yolo --epochs 80` |
| `inference.py` | single-image detection, counting and visualization |
| `inference_fullband.py` | tiled full-band scan; `--simulate` injects sources, `--fd-npz` scans a stored data set |
| `validate_fullband.py` | the eight simulated full-band scans behind the paper's full-band statements: `python -m yolo.validate_fullband` |
| `detection_metrics.py` | test-set true/false positives and misses, Extended Data Fig. 1: `python -m yolo.detection_metrics` |
| `detection_figures.py` | training curves, example detections, precision-recall curve |

Results of this release (released detector):

- test set, 500 images, IoU 0.3, confidence 0.25: UCB 4768 true positives, 91
  false positives, 489 misses (precision 0.981, recall 0.907); MBHB 254 true
  positives, 1 false positive, 4 misses (precision 0.996, recall 0.984);
- full-band scans (`validate_fullband.py`): 45 of 45 injected UCBs recovered
  over six 105-tile scans, largest frequency error 3.5 µHz, MBHB flagged in 98
  tiles at median confidence 0.96, noise-only scan empty, 2 detections without
  an injected counterpart in one scan; at the density ceiling 16 of 20 and 12
  of 13 sources inside one band are recovered.

The stage-2 and stage-3 networks use a frequency grid of T = 1/12 year; the
detector is applied to those data by placing them on its 15 s rfft grid
(`heron/stages.py::detect_yolo`).
