#!/usr/bin/env python3
"""
This 

Usage:
  python -m landmarks_only_training.src.evaluate_predictions_unet \
    --checkpoint /inwdata2a/sudhanshu/landmarks_only_training/outputs_unet_encoder_freezed/best_unet_landmarks.pt \
    --config landmarks_only_training/configs/default.yaml \
    --limit 20 \
    --visible_only \
    --visualize \
    --show_gt
"""

import argparse
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
from  landmarks_only_training.models.unet_encoder_model import EyeLandmarkUNetModel
from landmarks_only_training.models.backbones.unet_wrapper import UNetBackbone
from  landmarks_only_training.models.resnet_encoder_model import EyeLandmarkModel
from .utils import load_unet_encoder_backbone_from_ckpt

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

# Project path setup
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if __package__ in (None, ""):
    __package__ = "landmarks_only_training.src"

# Internal imports
from .utils import load_config, load_checkpoint
from .transforms import build_val_transforms



# ---------- Helpers ----------

def robust_parse_visibility(raw):
    if raw is None:
        return 0
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, (int, float)):
        try:
            if isinstance(raw, float) and torch.isnan(torch.tensor(raw)):
                return 0
        except Exception:
            pass
        return 1 if int(raw) != 0 else 0
    if not isinstance(raw, str):
        return 0
    t = raw.strip().lower().strip("\"' ")
    if t in ("true", "1", "yes", "y", "visible"):
        return 1
    if t in ("false", "0", "no", "n", "invisible", "nan", "none", "null", ""):
        return 0
    try:
        return 1 if int(t) != 0 else 0
    except ValueError:
        return 0

def parse_bbox(bbox_str):
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

def parse_landmarks_string(raw):
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
            pts.append((float(seg[0]), float(seg[1])))
        except ValueError:
            continue
    return pts

def denormalize_predictions(pred_tensor: torch.Tensor, bbox, cfg):
    """
    Convert predicted normalized local coords back to absolute face coords.
    Matches training normalization: width/height = (x2 - x1 + 1), (y2 - y1 + 1).
    """
    if bbox is None:
        return []
    x1, y1, x2, y2 = bbox
    bw = max(1.0, (x2 - x1 + 1))
    bh = max(1.0, (y2 - y1 + 1))
    normalized = cfg['data'].get('normalize_landmarks', False)
    local = cfg['data'].get('landmarks_local_coords', True)
    pts = []
    for i in range(pred_tensor.shape[0]):
        xn, yn = pred_tensor[i, 0].item(), pred_tensor[i, 1].item()
        if normalized and local:
            x_abs = xn * bw + x1
            y_abs = yn * bh + y1
        elif local and not normalized:
            x_abs = xn + x1
            y_abs = yn + y1
        else:
            x_abs = xn
            y_abs = yn
        pts.append((x_abs, y_abs))
    return pts

def visualize_sample(pil_img, pred_abs, gt_abs, out_png, show_gt=True):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(3, 3))
    plt.imshow(pil_img)
    plt.axis('off')
    handles = []
    if pred_abs:
        pred_handle = plt.scatter(
            [x for (x, y) in pred_abs], [y for (x, y) in pred_abs],
            c='lime', s=10, marker='o', edgecolors='black', linewidths=0.5, label='-pred'
        )
        handles.append(pred_handle)
    if show_gt and gt_abs:
        gt_handle = plt.scatter(
            [gx for (gx, gy) in gt_abs], [gy for (gx, gy) in gt_abs],
            c='red', s=10, marker='x', label='-gt'
        )
        handles.append(gt_handle)
    plt.title("Landmarks")
    if handles:
        plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches='tight')
    plt.close(fig)

def crop_eye_image(face_img: Image.Image, bbox, target_size: int) -> Image.Image:
    """
    Replicates training dataset crop logic:
      - Clamp bbox
      - Crop inclusive region (x2+1, y2+1)
      - Pad to square (center) if needed
      - Resize bilinear to target_size x target_size
    """
    if bbox is None:
        return None
    w_face, h_face = face_img.size
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), w_face - 1))
    x2 = max(0, min(int(x2), w_face - 1))
    y1 = max(0, min(int(y1), h_face - 1))
    y2 = max(0, min(int(y2), h_face - 1))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1

    crop = face_img.crop((x1, y1, x2 + 1, y2 + 1))
    cw, ch = crop.size
    if cw != ch:
        side = max(cw, ch)
        pad_w = side - cw
        pad_h = side - ch
        padded = Image.new("RGB", (side, side), (0, 0, 0))
        padded.paste(crop, (pad_w // 2, pad_h // 2))
        crop = padded
    if crop.size != (target_size, target_size):
        crop = crop.resize((target_size, target_size), Image.BILINEAR)
    return crop

# ---------- Args ----------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--visible_only", action="store_true")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--show_gt", action="store_true")
    ap.add_argument("--save_per_point", action="store_true",
                    help="Include per-point L2 and normalized distances in CSV.")
    ap.add_argument("--percentiles", type=str, default="90,95,99")
    ap.add_argument("--norm_mode", type=str, default="max_side",
                    choices=["max_side", "diagonal", "mean_side"],
                    help="Reference distance mode for NME: max_side | diagonal | mean_side")
    return ap.parse_args()

# ---------- Reference distance selection ----------

def compute_ref_distance(bw, bh, mode="max_side"):
    if mode == "max_side":
        return max(bw, bh)
    if mode == "diagonal":
        return (bw**2 + bh**2) ** 0.5
    if mode == "mean_side":
        return 0.5 * (bw + bh)
    return max(bw, bh)

# ---------- Main ----------

def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    overlay_dir = os.path.join(cfg["paths"]["output_dir"], "overlays")
    if args.visualize:
        os.makedirs(overlay_dir, exist_ok=True)

    report_txt = os.path.join(cfg["paths"]["output_dir"], "evaluation_landmarks.txt")
    per_sample_csv = os.path.join(cfg["paths"]["output_dir"], "per_sample_landmarks.csv")

    ckpt = load_checkpoint(args.checkpoint, map_location=device)



    if cfg["model"]["backbone"] == "resnet18":
        print(f"[INFO] (Eval) Loading ResNet18 backbone.")
        model = EyeLandmarkModel(
            hidden_landmarks=int(cfg['model']['hidden_landmarks']),
            dropout=float(cfg['model']['dropout']),
            backbone_name="resnet18"
        ).to(device)
        # load weights
        checkpoint = torch.load(args.checkpoint, map_location=device)
      
        # else, if checkpoint has 'state_dict' key
        model.load_state_dict(checkpoint['model_state'])

        # set model to evaluation mode if needed
        model.eval()

   



    elif cfg["model"]["backbone"] == "unet_encoder_unfreezed"   or cfg["model"]["backbone"] == "unet_encoder_freezed":
        if cfg["model"]["backbone"] == "unet_encoder_freezed":
            print(f"[INFO] (Eval) Loading pre-trained UNet encoder from: {cfg.get('model', {}).get('pretrain_encoder_ckpt', None)} (Freezed).")
        elif cfg["model"]["backbone"] == "unet_encoder_unfreezed":
            print(f"[INFO] (Eval) Loading Unet encoder backbone (unfreezed).")    
        backbone = UNetBackbone()
        backbone = backbone.to(device)  # ensure module is on device
        model = EyeLandmarkUNetModel(
            hidden_landmarks=int(cfg['model']['hidden_landmarks']),
            dropout=float(cfg['model']['dropout']),
            backbone=backbone
        ).to(device)

        # load weights
        checkpoint = torch.load(args.checkpoint, map_location=device)
      
        # else, if checkpoint has 'state_dict' key
        model.load_state_dict(checkpoint['model_state'])

        # set model to evaluation mode if needed
        model.eval()

    else:
        raise ValueError(f"Unknown backbone: {cfg['model']['backbone']}")    



    df = pd.read_csv(cfg['paths']['test_csv'])
    if args.limit is not None:
        df = df.head(args.limit)

    val_tf = build_val_transforms(cfg)
    target_size = int(cfg['data'].get('image_size', 128))

    per_sample_rows = []
    all_point_l2 = []
    all_point_norm = []
    per_image_nme = []
    samples_used = 0
    skipped = 0

    pct_list = [int(p.strip()) for p in args.percentiles.split(",") if p.strip().isdigit()]
    pck_thresholds =  cfg["inference"]["pck_thresholds"]

    for idx, row in enumerate(tqdm(df.itertuples(index=False), total=len(df), desc="Evaluating UNet landmarks")):
        img_path = getattr(row, "path_to_dataset", None)
        raw_vis = getattr(row, "eye_visibility", "true")
        gt_vis = robust_parse_visibility(raw_vis)
        if args.visible_only and gt_vis == 0:
            continue

        bbox = parse_bbox(getattr(row, "eye_bbox_face", ""))
        gt_lm_raw = getattr(row, "landmarks_coordinates_inside_eye_bbox", "")
        gt_lm = parse_landmarks_string(gt_lm_raw)

        if not isinstance(img_path, str) or not os.path.exists(img_path):
            skipped += 1
            continue
        if bbox is None or len(gt_lm) != cfg['data']['num_landmarks']:
            skipped += 1
            continue

        x1, y1, x2, y2 = bbox
        bw = max(1.0, (x2 - x1 + 1))
        bh = max(1.0, (y2 - y1 + 1))
        D_ref = compute_ref_distance(bw, bh, mode=args.norm_mode)

        face_img = Image.open(img_path).convert("RGB")
        eye_img = crop_eye_image(face_img, bbox, target_size)
        if eye_img is None:
            skipped += 1
            continue

        tensor_img = val_tf(eye_img).unsqueeze(0).to(device)
        # Convert to grayscale channel if RGB
        if tensor_img.dim() == 4 and tensor_img.size(1) == 3:
            tensor_img = tensor_img.mean(dim=1, keepdim=True)
        with torch.no_grad():
            out = model(tensor_img)
            pred_lm = out['landmarks'][0].cpu()

        pred_abs = denormalize_predictions(pred_lm, bbox, cfg)
        gt_abs = gt_lm

        pred_arr = np.array(pred_abs)
        gt_arr = np.array(gt_abs)
        point_l2 = np.sqrt(((pred_arr - gt_arr) ** 2).sum(axis=1))
        point_norm = point_l2 / D_ref

        all_point_l2.extend(point_l2.tolist())
        all_point_norm.extend(point_norm.tolist())

        nme_img = float(point_norm.mean())
        per_image_nme.append(nme_img)
        samples_used += 1

        row_dict = {
            "index": idx,
            "img_path": img_path,
            "gt_visibility": gt_vis,
            "mean_l2": float(point_l2.mean()),
            "mean_norm": nme_img,
            "D_ref": D_ref
        }
        if args.save_per_point:
            for i_pt, d in enumerate(point_l2):
                row_dict[f"l2_point_{i_pt}"] = float(d)
            for i_pt, dn in enumerate(point_norm):
                row_dict[f"norm_point_{i_pt}"] = float(dn)
        per_sample_rows.append(row_dict)

        if args.visualize:
            out_png = os.path.join(overlay_dir, f"sample_{idx}.png")
            visualize_sample(face_img, pred_abs, gt_abs, out_png, show_gt=args.show_gt)

    all_l2_arr = np.array(all_point_l2)
    all_norm_arr = np.array(all_point_norm)
    image_nme_arr = np.array(per_image_nme)

    if all_l2_arr.size == 0:
        summary = "No valid landmark samples evaluated."
    else:
        pixel_lines = [
            f"Samples evaluated: {samples_used}",
            f"Points evaluated: {all_l2_arr.size}",
            f"Mean L2 (px): {all_l2_arr.mean():.4f}",
            f"Std L2 (px): {all_l2_arr.std():.4f}",
            f"Median L2 (px): {np.median(all_l2_arr):.4f}",
        ]
        for p in pct_list:
            pixel_lines.append(f"{p}th percentile L2 (px): {np.percentile(all_l2_arr, p):.4f}")
        pixel_lines.append(f"Max L2 (px): {all_l2_arr.max():.4f}")

        nme_lines = [
            f"Mean NME (avg per image): {image_nme_arr.mean():.6f}",
            f"Std NME (per image): {image_nme_arr.std():.6f}",
            f"Median NME (per image): {np.median(image_nme_arr):.6f}",
        ]
        global_nme = all_norm_arr.mean()
        nme_lines.append(f"Global NME (all points mean norm): {global_nme:.6f}")

        pck_lines = []
        for thr in pck_thresholds:
            correct = (all_norm_arr <= thr).sum()
            total = all_norm_arr.size
            pck = correct / max(1, total)
            pck_lines.append(f"PCK@{thr:.3f}: {pck:.4f}")

        summary = "\n".join(pixel_lines + [""] + nme_lines + [""] + pck_lines)

    with open(report_txt, "w") as f:
        f.write("=== UNet Landmark Regression Evaluation ===\n")
        f.write(summary + "\n")
        f.write(f"Normalization mode: {args.norm_mode}\n")
        f.write(f"PCK thresholds: {pck_thresholds}\n")
        f.write(f"Skipped samples: {skipped}\n")

    pd.DataFrame(per_sample_rows).to_csv(per_sample_csv, index=False)

    print("\n=== UNet Landmark Regression Evaluation ===")
    print(summary)
    print(f"Normalization mode: {args.norm_mode}")
    print(f"PCK thresholds: {pck_thresholds}")
    print(f"Skipped samples: {skipped}")
    print("Per-sample CSV:", per_sample_csv)
    print("Report:", report_txt)
    if args.visualize:
        print("Overlays saved to:", overlay_dir)

if __name__ == "__main__":
    main()
