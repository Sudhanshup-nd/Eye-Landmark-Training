import argparse
import os
import torch
import cv2
import matplotlib.pyplot as plt
from .utils import load_config, ensure_dir, load_checkpoint
from .dataset import EyeDataset
from .transforms import build_val_transforms, resize_with_padding
from .model import EyeMultiTaskModel

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--limit", type=int, default=50)
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.output_dir)

    # Load model
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model = EyeMultiTaskModel(
        backbone_name=cfg['model']['backbone'],
        pretrained=False,
        hidden_landmarks=cfg['model']['hidden_landmarks'],
        dropout=cfg['model']['dropout']
    )
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    # Simple CSV read
    import pandas as pd
    df = pd.read_csv(args.csv)
    rows = df.head(args.limit)

    for i, row in rows.iterrows():
        path = row['eye_crop_path']
        if not os.path.exists(path):
            continue
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized, (ow, oh), (nw, nh), (pl, pt) = resize_with_padding(img, cfg['data']['image_size'], cfg['data']['keep_aspect_ratio'])

        from PIL import Image
        pil = Image.fromarray(resized)
        tf = build_val_transforms(cfg)
        tens = tf(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(tens)
            prob = torch.sigmoid(out['logits'])[0].item()
            preds = out['landmarks'][0].cpu().numpy().reshape(6,2)
        # Convert normalized coords back to resized space
        # We used original width/height for normalization; landmarks predicted in normalized space
        pred_pixels = []
        for (x_norm, y_norm) in preds:
            x_pix = x_norm * ow
            y_pix = y_norm * oh
            # scale to resized canvas:
            scale_x = nw / ow
            scale_y = nh / oh
            x_r = x_pix * scale_x + pl
            y_r = y_pix * scale_y + pt
            pred_pixels.append((x_r, y_r))

        fig = plt.figure(figsize=(3,3))
        plt.imshow(resized)
        title = f"VisProb={prob:.2f}"
        plt.title(title)
        if prob >= cfg['inference']['classification_threshold']:
            for (x,y) in pred_pixels:
                plt.scatter([x],[y], c='lime', s=20)
        else:
            plt.text(5,10,"Not Visible", color='red')
        plt.axis('off')
        out_path = os.path.join(args.output_dir, f"sample_{i}.png")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)

if __name__ == "__main__":
    main()