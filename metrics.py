import os

import numpy as np
from pathlib import Path

def read_yolo_boxes(txt_path, has_conf=False):
    """
    Reads YOLO .txt file.
    has_conf = True → file format: cls x y w h conf
    has_conf = False → cls x y w h
    Returns list of boxes: (cls, x, y, w, h[, conf])
    """
    boxes = []
    txt_path = Path(txt_path)
    if not txt_path.exists():
        return boxes

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            vals = list(map(float, parts[1:]))
            boxes.append((cls, *vals))
    return boxes


def xywh_to_xyxy(x, y, w, h):
    """Convert normalized center format → corner format (still normalized)."""
    return x - w / 2, y - h / 2, x + w / 2, y + h / 2


def iou_xyxy(boxA, boxB):
    """Compute IoU between two boxes in [x1, y1, x2, y2] normalized format."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = areaA + areaB - inter_area + 1e-12

    return inter_area / union


def compute_iou_file(gt_path, pred_path):
    """
    Compute IoU values between predicted boxes and ground-truth boxes
    for one image (one pair of .txt files).
    Returns list of IoUs (best GT match per prediction).
    """
    gt_boxes = read_yolo_boxes(gt_path, has_conf=False)
    pred_boxes = read_yolo_boxes(pred_path, has_conf=True)

    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return []

    ious = []
    for p in pred_boxes:
        cls_p, x_p, y_p, w_p, h_p, *conf = p
        p_xyxy = xywh_to_xyxy(x_p, y_p, w_p, h_p)

        best_iou = 0.0
        for g in gt_boxes:
            cls_g, x_g, y_g, w_g, h_g = g
            if cls_p != cls_g:  # match only same class
                continue
            g_xyxy = xywh_to_xyxy(x_g, y_g, w_g, h_g)
            best_iou = max(best_iou, iou_xyxy(p_xyxy, g_xyxy))

        ious.append(best_iou)

    return ious

def compute_mean_iou(gt_dir, pred_dir):
    """
    Compute IoU for all matching files between gt_dir and pred_dir.
    Returns mean IoU and per-image IoUs.
    """
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)
    assert gt_dir.exists(), f"GT folder not found: {gt_dir}"
    assert pred_dir.exists(), f"Pred folder not found: {pred_dir}"

    gt_files = list(gt_dir.glob("*.txt"))
    all_ious = []
    per_image = {}

    for gt_file in gt_files:
        pred_file = pred_dir / gt_file.name
        if not pred_file.exists():
            continue
        ious = compute_iou_file(gt_file, pred_file)
        if len(ious) > 0:
            per_image[gt_file.stem] = np.mean(ious)
            all_ious.extend(ious)

    if len(all_ious) == 0:
        print("⚠️  No IoUs computed (check filenames or empty files).")
        return 0.0, {}

    mean_iou = float(np.mean(all_ious))
    return mean_iou, per_image


if __name__ == "__main__":
    dir_gt = os.path.join("C:\\", "Users", "Utente", "DATASET", "CHALLENGE_2025_2026", "TEST", "LABELS", "G3")
    dir_pred = os.path.join("C:\\", "Users", "Utente", "DATASET", "CHALLENGE_2025_2026", "TEST", "LABELS", "TEMP", "G3")
    
    if os.path.exists(dir_gt) and os.path.exists(dir_pred):
        ious = compute_mean_iou(dir_gt, dir_pred)
        print(ious)
    else:
        print("Skipping metrics test: Paths not found on this machine.")