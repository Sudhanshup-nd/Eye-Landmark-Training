#!/usr/bin/env python3
"""
Visualize model predictions on samples from the NEW CSV format.

CSV columns expected:
    video_id, frame_key, eye_side, eye_visibility, path_to_dataset,
    eye_bbox_face, landmarks_coordinates_inside_eye_bbox

Assumes model was trained with:
    normalize_landmarks: True
    landmarks_local_coords: True

Usage:
    python -m src.visualize_samples \
        --checkpoint outputs/best_model.pt \
        --config configs/default.yaml \
        --csv data/val-v2.clean.csv \
        --output_dir outputs/vis_val \
        --limit 50 \
        --show_gt

Outputs PNG images with predicted landmarks (green) and optional GT (red).
Classification probability shown in title.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ---- Path patch to allow "python -m src.visualize_samples" and also direct script run ----
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if __package__ in (None, ""):
    __package__ = "src"
# ------------------------------------------------------------------------------------------

from src.utils import load_config, ensure_dir, load_checkpoint
from src.model import EyeMultiTaskModel
from src.transforms import build_val_transforms


def parse_bbox(bbox_str: str):
    if not isinstance(bbox_str, str):
        return None
    s = bbox_str.strip()
    if s.lower() in ("", "nan", "none", "null"):
        return None
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        return None
    try:
        x1, y1, x2, y2 = map(float, parts)
    except ValueError:
        return None
    return (x1, y1, x2, y2)


def parse_landmarks_string(raw: str):
    """
    Parse 'x,y;x,y;...' into list of (x,y).
    Returns [] if empty or invalid.
    """
    if not isinstance(raw, str):
        return []
    t = raw.strip()
    if t == "" or t.lower() in ("[]", "nan", "none", "null"):
        return []
    pts = []
    for token in t.split(";"):
        token = token.strip()
        if not token:
            continue
        seg = token.split(",")
        if len(seg) != 2:
            continue
        try:
            x = float(seg[0]); y = float(seg[1])
            pts.append((x, y))
        except ValueError:
            continue
    return pts


def denormalize_predictions(pred_tensor: torch.Tensor, bbox, cfg):
    """
    pred_tensor: shape [L,2], normalized local coords if training used normalization.
    bbox: (x1,y1,x2,y2)
    Returns list of (x_abs, y_abs) in original image pixel space.
    """
    if bbox is None:
        return []
    x1, y1, x2, y2 = bbox
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    # Model outputs normalized local coords if normalize_landmarks & landmarks_local_coords are True
    normalized = cfg['data'].get('normalize_landmarks', False)
    local = cfg['data'].get('landmarks_local_coords', True)

    pts = []
    for i in range(pred_tensor.shape[0]):
        xn, yn = pred_tensor[i, 0].item(), pred_tensor[i, 1].item()
        if normalized and local:
            x_abs = xn * bw + x1
            y_abs = yn * bh + y1
        elif local and not normalized:
            # Already local (absolute offsets)
            x_abs = xn + x1
            y_abs = yn + y1
        else:
            # Global absolute predicted? (not typical here)
            x_abs = xn
            y_abs = yn
        pts.append((x_abs, y_abs))
    return pts


def maybe_scale_for_display(image: Image.Image, points):
    """
    If you want to uniformly resize for display, do it here.
    Currently returns original image & points unchanged.
    """
    return image, points


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    ap.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    ap.add_argument("--csv", required=True, help="Path to CSV file")
    ap.add_argument("--output_dir", required=True, help="Output directory for visualizations")
    ap.add_argument("--limit", type=int, default=50, help="Number of samples to visualize")
    ap.add_argument("--show_gt", action="store_true", help="Overlay ground truth landmarks in red")
    ap.add_argument("--classification_threshold", type=float, default=None, help="Override probability threshold")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.output_dir)

    # Load checkpoint
    ckpt = load_checkpoint(args.checkpoint, map_location=device)

    model = EyeMultiTaskModel(
        backbone_name=cfg['model']['backbone'],
        pretrained=False,  # load weights from checkpoint, no need pretraining
        hidden_landmarks=cfg['model']['hidden_landmarks'],
        dropout=cfg['model']['dropout'],
        num_landmarks=cfg['data']['num_landmarks']
    )
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    # Classification threshold
    classification_threshold = args.classification_threshold
    if classification_threshold is None:
        classification_threshold = cfg.get('inference', {}).get('classification_threshold', 0.5)

    # Build val transforms (same as evaluation)
    val_tf = build_val_transforms(cfg)

    # Load CSV
    df = pd.read_csv(args.csv)
    rows = df.head(args.limit)

    for i, row in rows.iterrows():
        img_path = row.get("path_to_dataset")
        if not isinstance(img_path, str) or not os.path.exists(img_path):
            continue

        bbox = parse_bbox(row.get("eye_bbox_face", ""))
        gt_landmarks_raw = row.get("landmarks_coordinates_inside_eye_bbox", "")
        gt_landmarks = parse_landmarks_string(gt_landmarks_raw)

        # Load image
        pil_img = Image.open(img_path).convert("RGB")
        transformed = val_tf(pil_img)  # Tensor [C,H,W]

        with torch.no_grad():
            out = model(transformed.unsqueeze(0).to(device))
            prob = torch.sigmoid(out['logits'])[0].item()
            preds = out['landmarks'][0].cpu()  # [L,2]

        # De-normalize
        pred_abs = denormalize_predictions(preds, bbox, cfg)

        # Prepare figure
        fig = plt.figure(figsize=(3, 3))
        plt.imshow(pil_img)
        title = f"VisProb={prob:.2f}"
        plt.title(title)
        plt.axis('off')

        # Draw predicted landmarks if visible
        if prob >= classification_threshold:
            for (x, y) in pred_abs:
                plt.scatter([x], [y], c='lime', s=20, marker='o', edgecolors='black', linewidths=0.5)
        else:
            plt.text(5, 12, "Not Visible", color='red')

        # Draw GT landmarks (absolute) if requested
        if args.show_gt and bbox is not None and len(gt_landmarks) > 0:
            # If GT landmarks are absolute pixel coords (NOT normalized locally) – this matches your CSV
            for (gx, gy) in gt_landmarks:
                plt.scatter([gx], [gy], c='red', s=18, marker='x')

        out_file = os.path.join(args.output_dir, f"sample_{i}.png")
        plt.savefig(out_file, bbox_inches='tight')
        plt.close(fig)

    print(f"Saved {len(rows)} visualization images to {args.output_dir}")


if __name__ == "__main__":
    main()