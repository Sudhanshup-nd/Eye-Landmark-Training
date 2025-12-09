#!/usr/bin/env python3
"""
Split a single CSV (new schema) into train.csv and val.csv using an 80/20
split on unique video_id WITHOUT putting the same video_id in both splits.

Input CSV schema (new version):
video_id,frame_key,eye_side,eye_visibility,path_to_dataset,eye_bbox_face,landmarks_coordinates_inside_eye_bbox

Notes:
- eye_visibility is 'true'/'false' (case-insensitive)
- eye_bbox_face stored as "x1,y1,x2,y2"
- landmarks stored as "x,y;x,y;..." (semicolon separated pairs); may be empty or missing

USAGE:
  Edit the PATHS below and run:
    python scripts/split_train_val.py

Outputs:
  train.csv, val.csv in OUTPUT_DIR
"""

import pandas as pd
from pathlib import Path
import random

# ------------------ EDIT THESE PATHS ------------------
INPUT_CSV   = Path("/inwdata2a/sudhanshu/eye_multitask_training/outputs/eye_landmarks.csv")  # your full combined CSV path
OUTPUT_DIR  = Path("/inwdata2a/sudhanshu/eye_multitask_training/outputs/data")
TRAIN_OUT   = OUTPUT_DIR / "train-v2.csv"
VAL_OUT     = OUTPUT_DIR / "val-v2.csv"

TRAIN_RATIO = 0.80        # 80% train, 20% val
RANDOM_SEED = 42          # reproducible split
MIN_VAL_VIDEOS = 1        # ensure at least this many videos in val
# ------------------------------------------------------


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    if "video_id" not in df.columns:
        raise ValueError("Expected column 'video_id' not found in CSV.")

    # Get unique video IDs
    video_ids = sorted(df["video_id"].unique())
    if len(video_ids) == 0:
        raise ValueError("No video_id values found in CSV.")

    # Shuffle
    random.seed(RANDOM_SEED)
    random.shuffle(video_ids)

    # Compute split count
    n_total = len(video_ids)
    n_train_target = int(n_total * TRAIN_RATIO)
    
    # Ensure at least MIN_VAL_VIDEOS in validation
    n_train = max(0, min(n_train_target, n_total - MIN_VAL_VIDEOS))
    n_val   = n_total - n_train

    train_video_ids = set(video_ids[:n_train])
    val_video_ids   = set(video_ids[n_train:])

    # Filter rows
    train_df = df[df["video_id"].isin(train_video_ids)].copy()
    val_df   = df[df["video_id"].isin(val_video_ids)].copy()

    # Sanity checks
    overlap = train_video_ids.intersection(val_video_ids)
    if overlap:
        raise RuntimeError(f"Overlap in video IDs between train and val: {overlap}")

    # Save
    train_df.to_csv(TRAIN_OUT, index=False)
    val_df.to_csv(VAL_OUT, index=False)

    print(f"Total unique videos: {n_total}")
    print(f"Train videos: {len(train_video_ids)}  Val videos: {len(val_video_ids)}")
    print(f"Train rows: {len(train_df)}  Val rows: {len(val_df)}")
    print(f"Train CSV: {TRAIN_OUT}")
    print(f"Val CSV:   {VAL_OUT}")
    print("Done.")


if __name__ == "__main__":
    main()