"""
LISA Source Detection - YOLO Inference & Evaluation

Detect UCBs and SMBHBs in narrow-band LISA spectrograms.

Usage:
    # Evaluate on test set
    python -m yolo.inference --model weights/yolo_lisa_best.pt --dataset data/yolo --split test

    # Single image
    python -m yolo.inference --model weights/yolo_lisa_best.pt --image data/yolo/images/test/000042.png
"""

import numpy as np
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Dict, Optional
import json
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import *


class LISADetector:
    """Detect UCB and SMBHB sources in LISA spectrograms."""

    def __init__(self, model_path: str, conf: float = CONF_THRESHOLD,
                 iou: float = IOU_THRESHOLD):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.class_names = CLASS_NAMES

    def detect(self, image_path: str) -> Dict:
        """
        Detect sources in a spectrogram image.

        Returns:
            dict with 'ucb' and 'smbhb' lists of detections.
            Each detection has: confidence, x_center, y_center, width, height, bbox_xyxy.
            UCB detections also have frequency_hz, frequency_mhz.
        """
        results = self.model(image_path, conf=self.conf, iou=self.iou, verbose=False)[0]

        detections = {'ucb': [], 'smbhb': []}

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.class_names[cls_id]

            det = {
                'class': cls_name,
                'confidence': float(box.conf[0]),
                'x_center': float(box.xywhn[0][0]),
                'y_center': float(box.xywhn[0][1]),
                'width': float(box.xywhn[0][2]),
                'height': float(box.xywhn[0][3]),
                'bbox_xyxy': box.xyxy[0].cpu().numpy().tolist(),
            }

            if cls_name == 'ucb':
                det['frequency_hz'] = self._y_to_freq(det['y_center'])
                det['frequency_mhz'] = det['frequency_hz'] * 1e3

            detections[cls_name].append(det)

        return detections

    def _y_to_freq(self, y_center: float) -> float:
        freq_norm = 1.0 - y_center
        return F_MIN + freq_norm * (F_MAX - F_MIN)

    def count(self, image_path: str):
        dets = self.detect(image_path)
        return len(dets['ucb']), len(dets['smbhb'])

    def visualize(self, image_path: str, save_path: Optional[str] = None,
                  show: bool = True):
        img = np.array(Image.open(image_path))
        results = self.model(image_path, conf=self.conf, iou=self.iou, verbose=False)[0]

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(img)

        colors = {'ucb': 'cyan', 'smbhb': 'magenta'}
        n_ucb, n_smbhb = 0, 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.class_names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            color = colors[cls_name]
            rect = Rectangle((x1, y1), x2-x1, y2-y1,
                              fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1-5, f'{cls_name} {conf:.2f}', color=color, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

            if cls_name == 'ucb':
                n_ucb += 1
            else:
                n_smbhb += 1

        ax.set_title(f'Detected: {n_ucb} UCBs (cyan), {n_smbhb} SMBHBs (magenta)')
        ax.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        return fig


def load_labels(label_path: Path) -> Dict:
    labels = {'ucb': [], 'smbhb': []}
    if not label_path.exists():
        return labels
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
                labels[CLASS_NAMES[cls_id]].append({
                    'class': CLASS_NAMES[cls_id],
                    'x_center': x, 'y_center': y, 'width': w, 'height': h,
                })
    return labels


def evaluate(model_path: str, dataset_dir: str = 'data', split: str = 'test',
             conf: float = CONF_THRESHOLD, iou: float = IOU_THRESHOLD):
    """Evaluate YOLO detection performance."""
    detector = LISADetector(model_path, conf=conf, iou=iou)

    images_dir = Path(dataset_dir) / 'images' / split
    labels_dir = Path(dataset_dir) / 'labels' / split

    results = {
        'ucb': {'true': [], 'pred': []},
        'smbhb': {'true': [], 'pred': []},
    }

    for img_path in tqdm(sorted(images_dir.glob('*.png')), desc=f"Eval [{split}]"):
        label_path = labels_dir / f'{img_path.stem}.txt'
        gt = load_labels(label_path)
        preds = detector.detect(str(img_path))

        results['ucb']['true'].append(len(gt['ucb']))
        results['ucb']['pred'].append(len(preds['ucb']))
        results['smbhb']['true'].append(len(gt['smbhb']))
        results['smbhb']['pred'].append(len(preds['smbhb']))

    for cls in ['ucb', 'smbhb']:
        true = np.array(results[cls]['true'])
        pred = np.array(results[cls]['pred'])
        errors = pred - true
        print(f"\n{cls.upper()}:")
        print(f"  Count MAE: {np.mean(np.abs(errors)):.2f}")
        print(f"  Exact match: {np.mean(true == pred)*100:.1f}%")
        if cls == 'smbhb':
            tp = np.sum((true > 0) & (pred > 0))
            fn = np.sum((true > 0) & (pred == 0))
            fp = np.sum((true == 0) & (pred > 0))
            prec = tp / (tp + fp + 1e-6)
            rec = tp / (tp + fn + 1e-6)
            print(f"  Precision: {prec:.3f}  Recall: {rec:.3f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='data')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--conf', type=float, default=CONF_THRESHOLD)
    parser.add_argument('--image', type=str, default=None)
    args = parser.parse_args()

    if args.image:
        det = LISADetector(args.model, conf=args.conf)
        dets = det.detect(args.image)
        print(f"UCBs: {len(dets['ucb'])}, SMBHBs: {len(dets['smbhb'])}")
        det.visualize(args.image)
    else:
        evaluate(args.model, args.dataset, args.split, args.conf)
