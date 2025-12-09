# #!/usr/bin/env python3
# """
# Eye Landmark Training Script (module execution version)

# Run:
#     python -m src.train --config /inwdata2a/sudhanshu/landmarks_only_training/configs/default.yaml

# This trains a landmark regression model on cropped eye regions derived from
# face crop images. See earlier version header for detailed pipeline notes.
# """
# import argparse
# import os
# import torch
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# from .utils import load_config, seed_everything, ensure_dir, save_checkpoint
# from .dataset import EyeDataset
# from .transforms import build_train_transforms, build_val_transforms
# from .metrics import landmark_metrics, pck_metric
# from .model import EyeLandmarkModel
# from .losses import build_landmark_loss


# def parse_args():
#     ap = argparse.ArgumentParser(description="Train eye landmark regression model.")
#     ap.add_argument("--config", required=True, help="Path to YAML configuration file.")
#     return ap.parse_args()


# def summarize_config(cfg, dbg_enabled):
#     if not dbg_enabled:
#         return
#     print("[DEBUG] Config:")
#     def rec(d, indent=""):
#         for k in sorted(d.keys()):
#             v = d[k]
#             if isinstance(v, dict):
#                 print(f"[DEBUG] {indent}{k}:")
#                 rec(v, indent + "  ")
#             else:
#                 print(f"[DEBUG] {indent}{k}: {v}")
#     rec(cfg)


# def get_scheduler(optimizer, cfg):
#     sch = cfg['training'].get('lr_scheduler', 'none')
#     if sch == 'cosine':
#         from torch.optim.lr_scheduler import CosineAnnealingLR
#         return CosineAnnealingLR(optimizer, T_max=cfg['training']['epochs'])
#     return None


# def main():
#     args = parse_args()
#     cfg = load_config(args.config)
#     dbg_enabled = bool(cfg.get("debug", {}).get("enabled", False))
#     summarize_config(cfg, dbg_enabled)

#     seed_everything(cfg['seed'], dbg_enabled)
#     ensure_dir(cfg['paths']['output_dir'])
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if dbg_enabled:
#         print("[DEBUG] Using device:", device)

#     train_tf = build_train_transforms(cfg)  # Applies augmentations, converts images to tensors, and normalizes input data
#     val_tf   = build_val_transforms(cfg)

#     train_ds = EyeDataset(cfg['paths']['train_csv'], cfg, transform=train_tf, is_train=True)
#     val_ds   = EyeDataset(cfg['paths']['val_csv'], cfg, transform=val_tf, is_train=False)

#     train_loader = DataLoader(
#         train_ds,
#         batch_size=cfg['training']['batch_size'],
#         shuffle=True,
#         num_workers=cfg['training']['num_workers'],
#         pin_memory=True
#     )
#     val_loader = DataLoader(
#         val_ds,
#         batch_size=cfg['training']['batch_size'],
#         shuffle=False,
#         num_workers=cfg['training']['num_workers'],
#         pin_memory=True
#     )

#     model = EyeLandmarkModel(
#         backbone_name=cfg['model']['backbone'],
#         pretrained=cfg['model']['pretrained'],
#         hidden_landmarks=cfg['model']['hidden_landmarks'],
#         dropout=cfg['model']['dropout'],
#         num_landmarks=cfg['data']['num_landmarks']
#     ).to(device)

#     if dbg_enabled:
#         total_params = sum(p.numel() for p in model.parameters())
#         trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         print(f"[DEBUG] total_params={total_params} trainable={trainable}")

#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=cfg['training']['lr'],
#         weight_decay=cfg['training']['weight_decay']
#     )
#     scheduler = get_scheduler(optimizer, cfg)
#     loss_fn = build_landmark_loss(cfg)

#     ckpt_best = os.path.join(cfg['paths']['output_dir'], "best_landmarks.pt")
#     ckpt_last = os.path.join(cfg['paths']['output_dir'], "checkpoint_last_landmarks.pt")
#     best_metric_value = float('inf')
#     patience = cfg['training']['early_stop_patience']
#     epochs_no_improve = 0

#     for epoch in range(cfg['training']['epochs']):
#         # ---------------- TRAIN ----------------
#         model.train()
#         train_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
#         running_loss = 0.0
#         steps = 0
#         for batch in train_bar:
#             img  = batch['image'].to(device)
#             vis  = batch['visibility'].to(device)
#             lmk  = batch['landmarks'].to(device)
#             mask = batch['mask'].to(device)

#             out = model(img)
#             pred = out['landmarks']
#             lmk_loss, _ = loss_fn(pred, lmk, mask, vis)

#             optimizer.zero_grad()
#             lmk_loss.backward()
#             optimizer.step()

#             running_loss += lmk_loss.item()
#             steps += 1
#             train_bar.set_postfix({"loss": f"{lmk_loss.item():.4f}"})

#         avg_train_loss = running_loss / max(1, steps)

#         # ---------------- VALIDATION ----------------
#         model.eval()
#         all_pred, all_gt, all_mask, all_vis = [], [], [], []
#         with torch.no_grad():
#             for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
#                 img  = batch['image'].to(device)
#                 vis  = batch['visibility'].to(device)
#                 lmk  = batch['landmarks'].to(device)
#                 mask = batch['mask'].to(device)

#                 out  = model(img)
#                 all_pred.append(out['landmarks'])
#                 all_gt.append(lmk)
#                 all_mask.append(mask)
#                 all_vis.append(vis)

#         pred_cat = torch.cat(all_pred, 0)
#         gt_cat   = torch.cat(all_gt, 0)
#         mask_cat = torch.cat(all_mask, 0)
#         vis_cat  = torch.cat(all_vis, 0).view(-1)

#         eff_mask_val = mask_cat * vis_cat.unsqueeze(1)
#         eff_mask_exp = eff_mask_val.unsqueeze(-1)
#         valid_coords_val = (eff_mask_exp.sum() * 2.0).item()

#         abs_diff = torch.abs(pred_cat - gt_cat) * eff_mask_exp
#         val_lmk_l1 = abs_diff.sum().item() / max(valid_coords_val, 1e-6)

#         lmk_m = landmark_metrics(pred_cat, gt_cat, mask_cat)
#         pck_m = pck_metric(pred_cat, gt_cat, mask_cat, threshold=cfg['inference']['pck_threshold'])

#         print(f"[Epoch {epoch}] train_loss={avg_train_loss:.5f} "
#               f"val_L1={val_lmk_l1:.5f} NME={lmk_m['nme']:.4f} "
#               f"PCK={pck_m['pck']:.3f} valid_coords={int(valid_coords_val)}")

#         # ---------------- IMPROVEMENT / EARLY STOP STATUS ----------------
#         previous_best = best_metric_value
#         improved = val_lmk_l1 < best_metric_value
#         if improved:
#             best_metric_value = val_lmk_l1
#             save_checkpoint({
#                 "epoch": epoch,
#                 "model": model.state_dict(),
#                 "optimizer": optimizer.state_dict(),
#                 "best_metric": best_metric_value,
#                 "best_metric_name": "val_lmk_l1"
#             }, ckpt_best)
#             epochs_no_improve = 0
#             print(f"[INFO] val_L1 improved: {previous_best:.5f} -> {best_metric_value:.5f}. "
#                   f"Best model updated at: {ckpt_best}")
#         else:
#             epochs_no_improve += 1
#             print(f"[INFO] val_L1 did not improve (current {val_lmk_l1:.5f} >= best {previous_best:.5f}). "
#                   f"Early stopping counter: {epochs_no_improve}/{patience}")

#         # Always save last checkpoint
#         save_checkpoint({
#             "epoch": epoch,
#             "model": model.state_dict(),
#             "optimizer": optimizer.state_dict(),
#             "best_metric": best_metric_value,
#             "best_metric_name": "val_lmk_l1"
#         }, ckpt_last)

#         if scheduler:
#             scheduler.step()

#         if epochs_no_improve >= patience:
#             print(f"[INFO] Early stopping triggered after {patience} epochs without improvement.")
#             break

#     print("\n================= Training Complete =================")
#     print(f"Best val_L1:        {best_metric_value:.5f}")
#     print(f"Best model path:    {ckpt_best}")
#     print(f"Last checkpoint path: {ckpt_last}")
#     print("=====================================================")


# if __name__ == "__main__":
#     main()







#!/usr/bin/env python3
"""
Eye Landmark Training Script (module execution version)

Run:
    python -m src.train --config /inwdata2a/sudhanshu/landmarks_only_training/configs/default.yaml

This trains a landmark regression model on cropped eye regions derived from
face crop images. Includes landmark-aware augmentation pipeline in cropped space.
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import load_config, seed_everything, ensure_dir, save_checkpoint
from .dataset import EyeDataset
from .transforms import build_train_transforms, build_val_transforms
from .metrics import landmark_metrics, pck_metric
from .model import EyeLandmarkModel
from .losses import build_landmark_loss


def parse_args():
    ap = argparse.ArgumentParser(description="Train eye landmark regression model.")
    ap.add_argument("--config", required=True, help="Path to YAML configuration file.")
    return ap.parse_args()


def summarize_config(cfg, dbg_enabled):
    if not dbg_enabled:
        return
    print("[DEBUG] Config:")
    def rec(d, indent=""):
        for k in sorted(d.keys()):
            v = d[k]
            if isinstance(v, dict):
                print(f"[DEBUG] {indent}{k}:")
                rec(v, indent + "  ")
            else:
                print(f"[DEBUG] {indent}{k}: {v}")
    rec(cfg)


def get_scheduler(optimizer, cfg):
    sch = cfg['training'].get('lr_scheduler', 'none')
    if sch == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=cfg['training']['epochs'])
    return None


def main():
    args = parse_args()
    cfg = load_config(args.config)
    dbg_enabled = bool(cfg.get("debug", {}).get("enabled", False))
    summarize_config(cfg, dbg_enabled)

    seed_everything(cfg['seed'], dbg_enabled)
    ensure_dir(cfg['paths']['output_dir'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dbg_enabled:
        print("[DEBUG] Using device:", device)

    train_tf = build_train_transforms(cfg)
    val_tf   = build_val_transforms(cfg)

    train_ds = EyeDataset(cfg['paths']['train_csv'], cfg, transform=train_tf, is_train=True)
    val_ds   = EyeDataset(cfg['paths']['val_csv'], cfg, transform=val_tf, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True
    )

    model = EyeLandmarkModel(
        backbone_name=cfg['model']['backbone'],
        pretrained=cfg['model']['pretrained'],
        hidden_landmarks=cfg['model']['hidden_landmarks'],
        dropout=cfg['model']['dropout'],
        num_landmarks=cfg['data']['num_landmarks']
    ).to(device)

    if dbg_enabled:
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[DEBUG] total_params={total_params} trainable={trainable}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay']
    )
    scheduler = get_scheduler(optimizer, cfg)
    loss_fn = build_landmark_loss(cfg)

    ckpt_best = os.path.join(cfg['paths']['output_dir'], "best_landmarks.pt")
    ckpt_last = os.path.join(cfg['paths']['output_dir'], "checkpoint_last_landmarks.pt")
    best_metric_value = float('inf')
    patience = cfg['training']['early_stop_patience']
    epochs_no_improve = 0

    for epoch in range(cfg['training']['epochs']):
        # ---------------- TRAIN ----------------
        model.train()
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        running_loss = 0.0
        steps = 0
        for batch in train_bar:
            img  = batch['image'].to(device)
            vis  = batch['visibility'].to(device)
            lmk  = batch['landmarks'].to(device)
            mask = batch['mask'].to(device)

            out = model(img)
            pred = out['landmarks']
            lmk_loss, _ = loss_fn(pred, lmk, mask, vis)

            optimizer.zero_grad()
            lmk_loss.backward()
            optimizer.step()

            running_loss += lmk_loss.item()
            steps += 1
            train_bar.set_postfix({"loss": f"{lmk_loss.item():.4f}"})

        avg_train_loss = running_loss / max(1, steps)

        # ---------------- VALIDATION ----------------
        model.eval()
        all_pred, all_gt, all_mask, all_vis = [], [], [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                img  = batch['image'].to(device)
                vis  = batch['visibility'].to(device)
                lmk  = batch['landmarks'].to(device)
                mask = batch['mask'].to(device)

                out  = model(img)
                all_pred.append(out['landmarks'])
                all_gt.append(lmk)
                all_mask.append(mask)
                all_vis.append(vis)

        pred_cat = torch.cat(all_pred, 0)
        gt_cat   = torch.cat(all_gt, 0)
        mask_cat = torch.cat(all_mask, 0)
        vis_cat  = torch.cat(all_vis, 0).view(-1)

        eff_mask_val = mask_cat * vis_cat.unsqueeze(1)
        eff_mask_exp = eff_mask_val.unsqueeze(-1)
        valid_coords_val = (eff_mask_exp.sum() * 2.0).item()

        abs_diff = torch.abs(pred_cat - gt_cat) * eff_mask_exp
        val_lmk_l1 = abs_diff.sum().item() / max(valid_coords_val, 1e-6)

        lmk_m = landmark_metrics(pred_cat, gt_cat, mask_cat)
        pck_m = pck_metric(pred_cat, gt_cat, mask_cat, threshold=cfg['inference']['pck_threshold'])

        print(f"[Epoch {epoch}] train_loss={avg_train_loss:.5f} "
              f"val_L1={val_lmk_l1:.5f} NME={lmk_m['nme']:.4f} "
              f"PCK={pck_m['pck']:.3f} valid_coords={int(valid_coords_val)}")

        # ---------------- IMPROVEMENT / EARLY STOP STATUS ----------------
        previous_best = best_metric_value
        improved = val_lmk_l1 < best_metric_value
        if improved:
            best_metric_value = val_lmk_l1
            save_checkpoint({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_metric": best_metric_value,
                "best_metric_name": "val_lmk_l1"
            }, ckpt_best)
            epochs_no_improve = 0
            print(f"[INFO] val_L1 improved: {previous_best:.5f} -> {best_metric_value:.5f}. "
                  f"Best model updated at: {ckpt_best}")
        else:
            epochs_no_improve += 1
            print(f"[INFO] val_L1 did not improve (current {val_lmk_l1:.5f} >= best {previous_best:.5f}). "
                  f"Early stopping counter: {epochs_no_improve}/{patience}")

        # Always save last checkpoint
        save_checkpoint({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_metric": best_metric_value,
            "best_metric_name": "val_lmk_l1"
        }, ckpt_last)

        if scheduler:
            scheduler.step()

        if epochs_no_improve >= patience:
            print(f"[INFO] Early stopping triggered after {patience} epochs without improvement.")
            break

    print("\n================= Training Complete =================")
    print(f"Best val_L1:        {best_metric_value:.5f}")
    print(f"Best model path:    {ckpt_best}")
    print(f"Last checkpoint path: {ckpt_last}")
    print("=====================================================")


if __name__ == "__main__":
    main()