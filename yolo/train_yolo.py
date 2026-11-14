"""
Fine-tune the spectrogram detector (stage 1) from the pretrained base checkpoint.

    python -m yolo.dataset_generator --output data/yolo          # 5000 / 1000 / 500 images
    python -m yolo.train_yolo --data-dir data/yolo --epochs 80   # ~1.5 h on one GPU

The published detector (weights/yolo_lisa_best.pt) is the best-validation
checkpoint of an 80-epoch run with batch 16, image size 640, patience 20,
seed 0 and deterministic training, starting from weights/yolo26n_base.pt. The
full list of training arguments and the per-epoch metrics of that run are kept
in docs/provenance/yolo/ (args.yaml, results.csv).

The best checkpoint is copied to <project>/yolo_lisa_best.pt.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from heron.artifacts import YOLO_BASE                               # noqa: E402
from yolo.yolo_config import (IMG_SIZE, BATCH_SIZE, EPOCHS, PATIENCE,     # noqa: E402
                              TRAIN_SEED as SEED)


def create_dataset_yaml(data_dir: str) -> str:
    """Write dataset.yaml with the absolute path of this dataset copy."""
    data_dir = Path(data_dir).resolve()
    yaml_path = data_dir / 'dataset.yaml'
    yaml_path.write_text(
        f"path: {data_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"names:\n"
        f"  0: ucb\n"
        f"  1: smbhb\n\n"
        f"nc: 2\n"
    )
    return str(yaml_path)


def train(data_dir='data/yolo', epochs=EPOCHS, batch_size=BATCH_SIZE,
          img_size=IMG_SIZE, patience=PATIENCE, pretrained=YOLO_BASE,
          device=None, project='runs/train/yolo', name='yolo_lisa', seed=SEED,
          workers=8):
    """Fine-tune the detector; returns the path of the best checkpoint."""
    from ultralytics import YOLO
    yaml_path = create_dataset_yaml(data_dir)
    project = Path(project).resolve()
    print(f"base checkpoint: {pretrained}")
    model = YOLO(pretrained)
    model.train(data=yaml_path, epochs=epochs, batch=batch_size, imgsz=img_size,
                patience=patience, device=device, project=str(project), name=name,
                exist_ok=True, seed=seed, deterministic=True, workers=workers,
                save=True, plots=True, verbose=True)
    best = project / name / 'weights' / 'best.pt'
    dst = project / 'yolo_lisa_best.pt'
    if best.exists():
        shutil.copy2(best, dst)
        print(f"\nbest checkpoint -> {dst}")
    return dst


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--data-dir', type=str, default='data/yolo')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--img-size', type=int, default=IMG_SIZE)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument('--pretrained', type=str, default=YOLO_BASE,
                        help='base checkpoint to fine-tune from')
    parser.add_argument('--project', type=str, default='runs/train/yolo')
    parser.add_argument('--name', type=str, default='yolo_lisa')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', type=str, default=None,
                        help='e.g. "0" or "cpu" (default: first GPU if available)')
    args = parser.parse_args()
    train(args.data_dir, args.epochs, args.batch_size, img_size=args.img_size,
          patience=args.patience, pretrained=args.pretrained, device=args.device,
          project=args.project, name=args.name, seed=args.seed, workers=args.workers)
