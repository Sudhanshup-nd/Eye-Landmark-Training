
# in sudhanshu folder run 
# python -m landmarks_only_training.src.train_unet --config landmarks_only_training/configs/default.yaml

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .dataset import EyeDataset
from .transforms import build_train_transforms, build_val_transforms
from .losses import build_landmark_loss
from .utils import load_config, seed_everything, ensure_dir, save_checkpoint
from .metrics import landmark_metrics, pck_metric
from tqdm import tqdm
from .utils import load_unet_encoder_backbone_from_ckpt
from .new_unet_model import EyeLandmarkUNetModel
from .model import EyeLandmarkModel



def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        # Ensure grayscale for UNet backbone
        if images.dim() == 4 and images.size(1) == 3:
            images = images.mean(dim=1, keepdim=True)
        targets = batch["landmarks"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs["landmarks"], targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            if images.dim() == 4 and images.size(1) == 3:
                images = images.mean(dim=1, keepdim=True)
            targets = batch["landmarks"].to(device)
            outputs = model(images)
            loss = criterion(outputs["landmarks"], targets)
            total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)




def get_config(args):
    if not args.config:
        raise ValueError("--config path is required")
    return load_config(args.config)


def build_dataloaders(cfg):
    tfm_train = build_train_transforms(cfg)
    tfm_val = build_val_transforms(cfg)

    train_ds = EyeDataset(cfg['paths']['train_csv'], cfg, transform=tfm_train, is_train=True)
    val_ds = EyeDataset(cfg['paths']['val_csv'], cfg, transform=tfm_val, is_train=False)

    bs = int(cfg['training']['batch_size'])
    nw = int(cfg['training']['num_workers'])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    cfg = get_config(args)
    dbg_enabled = bool(cfg.get("debug", {}).get("enabled", False))
    seed_everything(cfg['seed'], dbg_enabled)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(cfg)

    # backbone = None
    # pretrain_path = cfg.get('model', {}).get('pretrain_encoder_ckpt', None)
    # if pretrain_path:
    #     print(f"[INFO] Loading pre-trained UNet encoder from: {pretrain_path}")
    #     backbone = load_unet_encoder_backbone_from_ckpt(pretrain_path, device=device)
    # else:
    #     print("[INFO] No pre-trained UNet encoder checkpoint provided; training from scratch.")


    # model = EyeLandmarkUNetModel(
    #     hidden_landmarks=int(cfg['model']['hidden_landmarks']),
    #     dropout=float(cfg['model']['dropout']),
    #     num_landmarks=int(cfg['data']['num_landmarks']),
    #     use_aux_head=False,
    #     backbone=backbone,
    # ).to(device)

    model = EyeLandmarkModel(
        backbone_name=cfg['model']['backbone'],
        pretrained=False,
        hidden_landmarks=cfg['model']['hidden_landmarks'],
        dropout=cfg['model']['dropout'],
        num_landmarks=cfg['data']['num_landmarks']
    ).to(device)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg['training']['lr']),
        weight_decay=float(cfg['training']['weight_decay'])
    )

    outdir = args.outdir or cfg['paths']['output_dir']
    ensure_dir(outdir, dbg=dbg_enabled)

    criterion = build_landmark_loss(cfg)

    ckpt_best = os.path.join(outdir, "best_unet_landmarks.pt")
    ckpt_last = os.path.join(outdir, "checkpoint_last_unet_landmarks.pt")
    best_metric_value = float('inf')
    patience = int(cfg['training']['early_stop_patience'])
    epochs_no_improve = 0
    epochs = int(cfg['training']['epochs'])

    for epoch in range(epochs):
        model.train()
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        running_loss = 0.0
        steps = 0
        for batch in train_bar:
            img  = batch['image'].to(device)
            vis  = batch['visibility'].to(device)
            lmk  = batch['landmarks'].to(device)
            mask = batch['mask'].to(device)

            if img.dim() == 4 and img.size(1) == 3:
                img = img.mean(dim=1, keepdim=True)

            out = model(img)
            pred = out['landmarks']
            lmk_loss, _ = criterion(pred, lmk, mask, vis)

            optimizer.zero_grad()
            lmk_loss.backward()
            optimizer.step()

            running_loss += lmk_loss.item()
            steps += 1
            train_bar.set_postfix({"loss": f"{lmk_loss.item():.4f}"})

        avg_train_loss = running_loss / max(1, steps)

        model.eval()
        all_pred, all_gt, all_mask, all_vis = [], [], [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                img  = batch['image'].to(device)
                vis  = batch['visibility'].to(device)
                lmk  = batch['landmarks'].to(device)
                mask = batch['mask'].to(device)

                if img.dim() == 4 and img.size(1) == 3:
                    img = img.mean(dim=1, keepdim=True)

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

        previous_best = best_metric_value
        improved = val_lmk_l1 < best_metric_value
        if improved:
            best_metric_value = val_lmk_l1
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_metric": best_metric_value,
                "best_metric_name": "val_lmk_l1"
            }, ckpt_best, dbg=dbg_enabled)
            epochs_no_improve = 0
            print(f"[INFO] val_L1 improved: {previous_best:.5f} -> {best_metric_value:.5f}. Best model updated at: {ckpt_best}")
        else:
            epochs_no_improve += 1
            print(f"[INFO] val_L1 did not improve (current {val_lmk_l1:.5f} >= best {previous_best:.5f}). "
                  f"Early stopping counter: {epochs_no_improve}/{patience}")

        save_checkpoint({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_metric": best_metric_value,
            "best_metric_name": "val_lmk_l1"
        }, ckpt_last, dbg=dbg_enabled)

        if epochs_no_improve >= patience:
            print(f"[INFO] Early stopping triggered after {patience} epochs without improvement.")
            break


if __name__ == "__main__":
    main()