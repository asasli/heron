"""
Detector test-set metrics (Supplementary Note "Stage-1 spectrogram detection",
Extended Data Fig. 1): predictions on the held-out test spectrograms are
matched to the ground-truth boxes (IoU >= 0.3, confidence >= 0.25) and counted
as true positives, false positives and misses per class.

    python -m yolo.detection_metrics            # 500 test images, ~1 min

Needs the detector test split (data/yolo/{images,labels}/test; docs/DATA.md).
Writes detection_summary.pdf (Extended Data Fig. 1) and the count figures to
results/figures/.
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from heron.artifacts import YOLO as YOLO_WEIGHTS, DATA, FIGURES   # noqa: E402

# Publication settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def iou(box1, box2):
    """Compute IoU between two boxes in YOLO format (x_center, y_center, w, h)"""
    # Convert to x1, y1, x2, y2
    x1_1 = box1[0] - box1[2] / 2
    y1_1 = box1[1] - box1[3] / 2
    x2_1 = box1[0] + box1[2] / 2
    y2_1 = box1[1] + box1[3] / 2
    
    x1_2 = box2[0] - box2[2] / 2
    y1_2 = box2[1] - box2[3] / 2
    x2_2 = box2[0] + box2[2] / 2
    y2_2 = box2[1] + box2[3] / 2
    
    # Intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    
    # Union
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def load_labels(label_path):
    """Load YOLO format labels"""
    labels = []
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    labels.append((cls, x, y, w, h))
    return labels


def compute_metrics(model_path, test_dir, labels_dir, iou_threshold=0.5, conf_threshold=0.25):
    """
    Compute detection metrics.
    
    Returns:
        metrics dict with TP, FP, FN, precision, recall per class
    """
    from ultralytics import YOLO
    
    model = YOLO(model_path)
    test_images = sorted(Path(test_dir).glob('*.png'))
    labels_dir = Path(labels_dir)
    
    # Counters per class
    class_names = ['UCB', 'SMBHB']
    n_classes = len(class_names)
    
    TP = defaultdict(int)  # True Positives
    FP = defaultdict(int)  # False Positives
    FN = defaultdict(int)  # False Negatives
    
    # Also track confusion between classes
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    
    print(f"Evaluating {len(test_images)} images...")
    
    for img_path in tqdm(test_images):
        # Get predictions
        results = model(str(img_path), verbose=False, conf=conf_threshold)
        
        pred_boxes = []
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xywhn = box.xywhn[0].cpu().numpy()
            pred_boxes.append((cls, xywhn[0], xywhn[1], xywhn[2], xywhn[3], conf))
        
        # Get ground truth
        label_path = labels_dir / f"{img_path.stem}.txt"
        gt_boxes = load_labels(label_path)
        
        # Match predictions to ground truth
        gt_matched = [False] * len(gt_boxes)
        
        for pred in pred_boxes:
            pred_cls = pred[0]
            pred_box = pred[1:5]
            
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_matched[gt_idx]:
                    continue
                    
                gt_cls = gt[0]
                gt_box = gt[1:5]
                
                iou_val = iou(pred_box, gt_box)
                
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                gt_matched[best_gt_idx] = True
                gt_cls = gt_boxes[best_gt_idx][0]
                
                if pred_cls == gt_cls:
                    # Correct detection
                    TP[pred_cls] += 1
                else:
                    # Wrong class
                    FP[pred_cls] += 1
                    FN[gt_cls] += 1
                    confusion[gt_cls, pred_cls] += 1
            else:
                # No matching GT - false positive
                FP[pred_cls] += 1
        
        # Unmatched GT boxes are false negatives
        for gt_idx, gt in enumerate(gt_boxes):
            if not gt_matched[gt_idx]:
                FN[gt[0]] += 1
    
    # Compute metrics
    metrics = {
        'class_names': class_names,
        'per_class': {},
        'confusion_matrix': confusion
    }
    
    for cls_idx, cls_name in enumerate(class_names):
        tp = TP[cls_idx]
        fp = FP[cls_idx]
        fn = FN[cls_idx]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics['per_class'][cls_name] = {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        print(f"\n{cls_name}:")
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1: {f1:.3f}")
    
    return metrics


def plot_detection_summary(metrics, output_dir):
    """
    Create a summary figure showing detection performance.
    """
    class_names = metrics['class_names']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Precision, Recall, F1 per class
    ax = axes[0]
    x = np.arange(len(class_names))
    width = 0.25
    
    precision = [metrics['per_class'][c]['precision'] for c in class_names]
    recall = [metrics['per_class'][c]['recall'] for c in class_names]
    f1 = [metrics['per_class'][c]['f1'] for c in class_names]
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2ecc71')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#3498db')
    bars3 = ax.bar(x + width, f1, width, label='F1 Score', color='#9b59b6')
    
    ax.set_ylabel('Score')
    ax.set_title('(a) Detection Performance by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: TP, FP, FN counts
    ax = axes[1]
    
    tp_counts = [metrics['per_class'][c]['TP'] for c in class_names]
    fp_counts = [metrics['per_class'][c]['FP'] for c in class_names]
    fn_counts = [metrics['per_class'][c]['FN'] for c in class_names]
    
    bars1 = ax.bar(x - width, tp_counts, width, label='True Positives', color='#27ae60')
    bars2 = ax.bar(x, fp_counts, width, label='False Positives', color='#e74c3c')
    bars3 = ax.bar(x + width, fn_counts, width, label='False Negatives', color='#f39c12')
    
    ax.set_ylabel('Count')
    ax.set_title('(b) Detection Counts by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                       f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / 'detection_summary.pdf'
    plt.savefig(output_path)
    plt.savefig(output_dir / 'detection_summary.png')
    plt.close()
    print(f"\nsaved: {output_path}")


def plot_simple_confusion_matrix(metrics, output_dir):
    """
    Plot a simple 2x2 confusion matrix for the two classes.
    Shows: correctly detected, missed, false alarms
    """
    class_names = metrics['class_names']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, cls_name in enumerate(class_names):
        ax = axes[idx]
        m = metrics['per_class'][cls_name]
        
        # 2x2 matrix: [Detected/Missed] vs [Actual Positive/Actual Negative proxy]
        tp = m['TP']
        fp = m['FP']
        fn = m['FN']
        
        # Simple visualization
        data = np.array([[tp, fn], [fp, 0]])
        
        # But actually let's do a pie chart for clarity
        ax.clear()
        
        total_gt = tp + fn
        total_pred = tp + fp
        
        # Bar chart instead
        categories = ['Correct\nDetections', 'Missed\n(FN)', 'False\nAlarms (FP)']
        values = [tp, fn, fp]
        colors = ['#27ae60', '#f39c12', '#e74c3c']
        
        bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Count')
        ax.set_title(f'{cls_name} Detection Results')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add percentages
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Add precision/recall annotation
        precision = m['precision']
        recall = m['recall']
        ax.text(0.98, 0.98, f'Precision: {precision:.1%}\nRecall: {recall:.1%}',
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=11)
    
    plt.tight_layout()
    output_path = output_dir / 'confusion_matrix.pdf'
    plt.savefig(output_path)
    plt.savefig(output_dir / 'confusion_matrix.png')
    plt.close()
    print(f"saved: {output_path}")


def plot_combined_metrics(metrics, output_dir):
    """
    Single publication-quality figure combining all metrics.
    """
    class_names = metrics['class_names']
    
    fig = plt.figure(figsize=(10, 8))
    
    # Create a 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Top: Per-class detection rates
    ax1 = fig.add_subplot(gs[0, :])
    
    x = np.arange(len(class_names))
    width = 0.35
    
    precision = [metrics['per_class'][c]['precision'] * 100 for c in class_names]
    recall = [metrics['per_class'][c]['recall'] * 100 for c in class_names]
    
    bars1 = ax1.bar(x - width/2, precision, width, label='Precision', color='#3498db', edgecolor='black')
    bars2 = ax1.bar(x + width/2, recall, width, label='Recall', color='#e74c3c', edgecolor='black')
    
    ax1.set_ylabel('Score (%)')
    ax1.set_title('Detection Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, fontsize=12)
    ax1.set_ylim(0, 110)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Bottom left: UCB breakdown
    ax2 = fig.add_subplot(gs[1, 0])
    m = metrics['per_class']['UCB']
    labels = ['Correct', 'Missed', 'False +']
    sizes = [m['TP'], m['FN'], m['FP']]
    colors = ['#27ae60', '#f39c12', '#e74c3c']
    explode = (0.05, 0, 0)
    
    if sum(sizes) > 0:
        wedges, texts, autotexts = ax2.pie(sizes, explode=explode, labels=labels, colors=colors,
                                            autopct='%1.0f%%', startangle=90,
                                            textprops={'fontsize': 10})
        ax2.set_title('UCB Detections')
    
    # Bottom right: SMBHB breakdown
    ax3 = fig.add_subplot(gs[1, 1])
    m = metrics['per_class']['SMBHB']
    sizes = [m['TP'], m['FN'], m['FP']]
    
    if sum(sizes) > 0:
        wedges, texts, autotexts = ax3.pie(sizes, explode=explode, labels=labels, colors=colors,
                                            autopct='%1.0f%%', startangle=90,
                                            textprops={'fontsize': 10})
        ax3.set_title('SMBHB Detections')
    
    plt.suptitle('LISA-YOLO Detection Performance Summary', fontsize=14, fontweight='bold', y=1.02)
    
    output_path = output_dir / 'detection_performance.pdf'
    plt.savefig(output_path)
    plt.savefig(output_dir / 'detection_performance.png')
    plt.close()
    print(f"saved: {output_path}")


def print_latex_table(metrics):
    """Print LaTeX table for paper."""
    print("\n" + "="*60)
    print("LATEX TABLE")
    print("="*60)
    
    print(r"""
\begin{table}[t]
\centering
\caption{Detection performance on the test set.}
\label{tab:detection_results}
\begin{tabular}{lccccc}
\toprule
Class & TP & FP & FN & Precision & Recall \\
\midrule""")
    
    for cls_name in metrics['class_names']:
        m = metrics['per_class'][cls_name]
        print(f"{cls_name} & {m['TP']} & {m['FP']} & {m['FN']} & "
              f"{m['precision']*100:.1f}\\% & {m['recall']*100:.1f}\\% \\\\")
    
    print(r"""\bottomrule
\end{tabular}
\end{table}
""")


def main():
    parser = argparse.ArgumentParser(description='Compute confusion matrix')
    parser.add_argument('--model', type=str, default=YOLO_WEIGHTS, help='Path to model')
    parser.add_argument('--test-dir', type=str, default=str(DATA / 'yolo' / 'images' / 'test'))
    parser.add_argument('--labels-dir', type=str, default=str(DATA / 'yolo' / 'labels' / 'test'))
    parser.add_argument('--output-dir', type=str, default=str(FIGURES), help='Output directory')
    parser.add_argument('--iou-threshold', type=float, default=0.3, help='IoU threshold for matching')
    parser.add_argument('--conf-threshold', type=float, default=0.25, help='Confidence threshold')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("COMPUTING DETECTION METRICS")
    print("="*60)
    
    metrics = compute_metrics(
        args.model, 
        args.test_dir, 
        args.labels_dir,
        iou_threshold=args.iou_threshold,
        conf_threshold=args.conf_threshold
    )
    
    print("\n" + "="*60)
    print("GENERATING FIGURES")
    print("="*60)
    
    plot_simple_confusion_matrix(metrics, output_dir)
    plot_detection_summary(metrics, output_dir)
    plot_combined_metrics(metrics, output_dir)
    
    print_latex_table(metrics)
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)


if __name__ == '__main__':
    main()
