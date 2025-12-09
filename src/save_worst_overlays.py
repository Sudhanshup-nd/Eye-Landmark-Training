"""
Save Worst Overlays
-------------------
Evaluate landmark regression predictions on the validation set, rank samples by
masked L1 error (optionally also report NME/PCK), and save overlay visualizations
for the worst X% cases.

Features:
1. Loads config + trained checkpoint.
2. Performs inference over validation samples (no gradient).
3. Computes masked L1 error per sample using (visibility * mask).
4. Optionally computes NME and PCK for reference.
5. Ranks all samples by error descending and selects top --percent%.
6. Saves overlays:
     a) Full frame with projected landmarks.
     b) Eye bounding-box crop with landmarks.
7. Optional CSV summary of worst cases.

Important Implementation Notes:
- The dataset returns transformed eye crops only; it does NOT provide the full-frame
    image tensor. We reconstruct full image paths using the underlying dataframe via
    dataset.valid_indices, assuming DataLoader shuffle=False.
- Landmark targets are normalized local coords in [0,1] relative to bounding box width/height.
    Thus denormalization uses (x2 - x1 + 1, y2 - y1 + 1) rather than crop tensor size.
- Bounding-box slicing here uses inclusive end (+1) matching dataset cropping semantics.

Usage:
        python -m src.save_worst_overlays \
                --config configs/default.yaml \
                --checkpoint outputs_landmarks/best_landmarks.pt \
                --percent 5 \
                --output_dir outputs_landmarks/eval/overlays_worst \
                --save_csv 1 \
                --pck_threshold 0.1

CSV Output Columns (if enabled):
        rank,error,video_id,frame_key,eye_side,image_path

Recommended Workflow:
1. Run training to produce `best_landmarks.pt`.
2. Execute this script to inspect difficult samples.
3. Review overlays and adjust data quality / augmentation / model accordingly.
"""
import os
import math
import argparse
from typing import Tuple, List, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .utils import load_config, seed_everything as set_seed
from .dataset import EyeDataset
from .transforms import build_val_transforms
from .model import EyeLandmarkModel


def draw_landmarks(img_rgb: np.ndarray, pts: np.ndarray, color=(0, 0, 255), radius: int = 2) -> np.ndarray:
    """
    Draw landmarks (pixel coordinates) on an RGB image.
    pts: array of shape [L, 2] with pixel coordinates (x, y).
    """
    out = img_rgb.copy()
    h, w = out.shape[:2]
    for x, y in pts:
        xi = int(np.clip(x, 0, w - 1))
        yi = int(np.clip(y, 0, h - 1))
        cv2.circle(out, (xi, yi), radius, color, -1)
    return out


def compute_l1_error(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    """Average masked L1 over coordinates (x,y) for valid landmarks only."""
    valid = mask.sum().item()
    if valid == 0:
        return 0.0
    diff = (pred - gt).abs() * mask.view(-1, 1)
    return diff.sum().item() / (valid * 2.0)

def compute_nme(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean Euclidean distance (NME style) over valid landmarks (local units)."""
    valid = mask.sum().item()
    if valid == 0:
        return 0.0
    dist = torch.sqrt(((pred - gt) ** 2).sum(-1) + 1e-8) * mask
    return dist.sum().item() / (valid + 1e-6)

def compute_pck(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, threshold: float) -> float:
    """Percentage of Correct Keypoints under local-threshold (normalized units)."""
    if mask.sum().item() == 0:
        return 0.0
    dist = torch.sqrt(((pred - gt) ** 2).sum(-1))
    hits = ((dist <= threshold) * mask).sum().item()
    total = mask.sum().item()
    return hits / (total + 1e-6)


def denorm_landmarks(norm_pts: torch.Tensor, bbox_xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    """Denormalize local landmark coords (in [0,1]) using bbox width/height."""
    x1, y1, x2, y2 = bbox_xyxy
    bw = max(1.0, float(x2 - x1 + 1))
    bh = max(1.0, float(y2 - y1 + 1))
    pts = norm_pts.detach().cpu().numpy().copy()
    pts[:, 0] *= bw
    pts[:, 1] *= bh
    return pts


def save_overlay(full_rgb: np.ndarray,
                 bbox_xyxy: Tuple[int, int, int, int],
                 pred_local_px: np.ndarray,
                 gt_local_px: np.ndarray,
                 out_dir_overlays: str,
                 base_name: str) -> None:
    """Save full-face overlays with both predictions and ground-truth landmarks.

    - Projects local bbox coords to full image and draws:
      - Predictions in blue circles
      - Ground truth in red crosses
    """
    x1, y1, x2, y2 = bbox_xyxy
    # Project to full-frame coordinates
    pred_full = pred_local_px.copy()
    pred_full[:, 0] += float(x1)
    pred_full[:, 1] += float(y1)
    gt_full = gt_local_px.copy()
    gt_full[:, 0] += float(x1)
    gt_full[:, 1] += float(y1)

    # Draw predictions (lime/green circles) then GT (red crosses), matching evaluate_predictions style
    full_pred = draw_landmarks(full_rgb, pred_full, color=(0, 255, 0), radius=2)
    # Draw GT on top
    out = full_pred.copy()
    h, w = out.shape[:2]
    for x, y in gt_full:
        xi = int(np.clip(x, 0, w - 1))
        yi = int(np.clip(y, 0, h - 1))
        cv2.drawMarker(out, (xi, yi), (255, 0, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=6, thickness=1)

    os.makedirs(out_dir_overlays, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir_overlays, f"{base_name}_full.jpg"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(description="Save worst-x% overlays based on landmark regression errors.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML.")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs_landmarks/best_landmarks.pt",
                        help="Path to trained model checkpoint.")
    parser.add_argument("--percent", type=float, default=5.0,
                        help="Top percentage of worst cases to save (e.g., 5 for top 5%).")
    parser.add_argument("--output_dir", type=str,
                        default="outputs_landmarks/eval/overlays_worst",
                        help="Directory to save worst-case overlays.")
    parser.add_argument("--save_csv", type=int, default=1, help="Whether to save CSV summary (1/0).")
    parser.add_argument("--pck_threshold", type=float, default=0.1, help="Threshold for local PCK calculation.")
    parser.add_argument("--debug", type=int, default=0, help="Enable shape debug prints (1/0).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset (validation)
    val_tf = build_val_transforms(cfg)
    val_ds = EyeDataset(cfg["paths"]["val_csv"], cfg, transform=val_tf, is_train=False)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True
    )

    # Model
    mdl = EyeLandmarkModel(
        backbone_name=cfg["model"]["backbone"],
        pretrained=cfg["model"]["pretrained"],
        hidden_landmarks=cfg["model"]["hidden_landmarks"],
        dropout=cfg["model"]["dropout"],
        num_landmarks=cfg["data"]["num_landmarks"],
    )
    chk = torch.load(args.checkpoint, map_location="cpu")
    state = chk.get("model", chk)
    mdl.load_state_dict(state, strict=False)
    mdl.to(device)
    mdl.eval()

    # Helper to safely fetch batch items
    def _safe(batch_dict, key, i, default=""):
        if key not in batch_dict:
            return default
        v = batch_dict[key]
        try:
            if isinstance(v, (list, tuple)):
                return v[i] if i < len(v) else default
            if torch.is_tensor(v):
                return v[i]
            return v
        except Exception:
            return default

    # Collect per-sample metrics & metadata
    records: List[Dict[str, any]] = []
    global_idx = 0  # maps batch sample to dataset.valid_indices

    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].to(device)               # [B,3,H,W]
            gt = batch["landmarks"].to(device)            # [B,L,2]
            mask = batch["mask"].to(device)               # [B,L]
            vis = batch["visibility"].to(device)          # [B] or [B,1]

            out = mdl(img)
            pred = out.get("landmarks")
            if pred is None:
                continue

            # Normalize visibility tensor to shape [B,1] then broadcast
            if vis.dim() == 1:
                vis_eff = vis.view(-1, 1)    # [B,1]
            elif vis.dim() == 2 and vis.size(1) == 1:
                vis_eff = vis                 # [B,1]
            else:
                # Unexpected shape (e.g., [B,L]); collapse to per-sample visibility by mean
                vis_eff = vis.mean(dim=1, keepdim=True)

            eff_mask = mask * vis_eff        # broadcast: [B,L] * [B,1] -> [B,L]

            if args.debug:
                print(f"[DEBUG] batch_size={img.size(0)} pred_shape={pred.shape} gt_shape={gt.shape} mask_shape={mask.shape} vis_shape={vis.shape} vis_eff_shape={vis_eff.shape}")

            B = pred.size(0)
            for i in range(B):
                # Map to original dataframe row using valid_indices
                if global_idx + i >= len(val_ds.valid_indices):
                    continue
                real_row_idx = val_ds.valid_indices[global_idx + i]
                row = val_ds.df.iloc[real_row_idx]

                bbox_tensor = batch["bbox"][i]
                bbox_np = bbox_tensor.detach().cpu().numpy()
                err_l1 = compute_l1_error(pred[i], gt[i], eff_mask[i])
                err_nme = compute_nme(pred[i], gt[i], eff_mask[i])
                err_pck = compute_pck(pred[i], gt[i], eff_mask[i], args.pck_threshold)
                if args.debug and i == 0:
                    print(f"[DEBUG] sample err_l1={err_l1:.4f} nme={err_nme:.4f} pck={err_pck:.3f} bbox={bbox_np.tolist()} vis_mean={vis_eff[i].item():.3f}")

                records.append({
                    "error": err_l1,
                    "error_nme": err_nme,
                    "error_pck": err_pck,
                    "pred_norm": pred[i].detach().cpu(),
                    "gt_norm": gt[i].detach().cpu(),
                    "bbox": bbox_np,
                    "image_path": row.path_to_dataset,
                    "video_id": row.video_id,
                    "frame_key": row.frame_key,
                    "eye_side": row.eye_side,
                })
            global_idx += B

    if len(records) == 0:
        print("No records computed; check dataset and model outputs.")
        return

    # Sort by error descending and take top X%
    records.sort(key=lambda r: r["error"], reverse=True)
    k = max(1, math.floor(len(records) * (args.percent / 100.0)))
    worst = records[:k]

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving {len(worst)} worst-case overlays to {args.output_dir} (top {args.percent}% of {len(records)})")

    csv_rows = []
    for rank, rec in enumerate(worst, start=1):
        image_path = rec["image_path"]
        bbox_xyxy = rec["bbox"].astype(np.int32)
        pred_norm = rec["pred_norm"]
        video_id = rec["video_id"]
        frame_key = rec["frame_key"]
        eye_side = rec["eye_side"]
        err = rec["error"]

        if not image_path or not os.path.exists(image_path):
            continue
        bgr = cv2.imread(image_path)
        if bgr is None:
            continue
        full_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        bbox_tuple = tuple(map(int, bbox_xyxy.tolist()))
        pred_local_px = denorm_landmarks(pred_norm, bbox_tuple)
        gt_local_px = denorm_landmarks(rec.get("gt_norm", pred_norm*0), bbox_tuple)
        base = f"rank{rank:04d}_err{err:.4f}_{video_id}_{frame_key}_{eye_side}"
        save_overlay(full_rgb, bbox_tuple, pred_local_px, gt_local_px, args.output_dir, base)

        if args.save_csv:
            csv_rows.append([
                rank,
                f"{err:.6f}",
                f"{rec['error_nme']:.6f}",
                f"{rec['error_pck']:.6f}",
                video_id,
                frame_key,
                eye_side,
                image_path,
            ])

    if args.save_csv and csv_rows:
        import csv
        csv_path = os.path.join(args.output_dir, "worst_overlays_summary.csv")
        header = ["rank", "l1_error", "nme", "pck", "video_id", "frame_key", "eye_side", "image_path"]
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(csv_rows)
        print(f"Saved CSV summary: {csv_path}")

    print("Done.")


if __name__ == "__main__":
    main()
