# import pandas as pd

# train = pd.read_csv("/inwdata2a/sudhanshu/landmarks_only_training/data/train-v2.clean.csv")
# val = pd.read_csv("/inwdata2a/sudhanshu/landmarks_only_training/data/val-v2.clean.csv")

# def show_stats(df, name):
#     vc = df['eye_visibility'].value_counts().to_dict()
#     print(f"{name}: rows={len(df)}, visibility_counts={vc}")

# show_stats(train, "TRAIN")
# show_stats(val, "VAL")

# train_vids = set(train['video_id'].unique())
# val_vids = set(val['video_id'].unique())
# overlap = train_vids & val_vids

# print(f"Unique video_ids: TRAIN={len(train_vids)} VAL={len(val_vids)} OVERLAP={len(overlap)}")
# if overlap:
#     # show first few overlapping ids
#     print("Sample overlapping video_ids:", list(overlap)[:10])








#!/usr/bin/env python3
"""
Check detailed statistics of the landmark CSV(s).

This version hard-codes the commonly used absolute paths so you can run it
without remembering them. You can still override with --csv arguments if needed.

What it reports (for each CSV):
  - Total rows
  - Strict visibility counts (True / False)
  - Counts of rows with valid bbox
  - Counts of rows with exactly EXPECTED_L landmarks
  - Joint counts:
      * visible & EXPECTED_L landmarks
      * invisible & EXPECTED_L landmarks
      * visible & valid bbox
      * invisible & valid bbox
  - Eye side distribution (left/right or others)
  - Lists (up to max_show) of any suspicious rows:
      * invisible but having EXPECTED_L landmarks
      * invisible but having a valid bbox

Adjust EXPECTED_L or hard-coded paths below as needed.

Usage:
    python scripts/check_csv_stats.py
    python scripts/check_csv_stats.py --csv /custom/path/to/val.csv --landmark_count 6
    python scripts/check_csv_stats.py --train_csv /custom/train.csv --val_csv /custom/val.csv

Hard-coded default paths:
    TRAIN_CSV_DEFAULT = /inwdata2a/sudhanshu/landmarks_only_training/data/train-v2.clean.csv
    VAL_CSV_DEFAULT   = /inwdata2a/sudhanshu/landmarks_only_training/data/val-v2.clean.csv
"""

import argparse
import pandas as pd
from collections import Counter
import sys
from pathlib import Path

# ------------------------------------------------------------------
# Hard-coded commonly used paths (edit here if your directory moves)
# ------------------------------------------------------------------
TRAIN_CSV_DEFAULT = "/inwdata2a/sudhanshu/landmarks_only_training/data/train-v2.clean.csv"
VAL_CSV_DEFAULT   = "/inwdata2a/sudhanshu/landmarks_only_training/data/val-v2.clean.csv"

EXPECTED_L_DEFAULT = 6  # number of landmarks expected

REQUIRED_COLS = [
    "video_id",
    "frame_key",
    "eye_side",
    "eye_visibility",
    "path_to_dataset",
    "eye_bbox_face",
    "landmarks_coordinates_inside_eye_bbox"
]

def parse_args():
    ap = argparse.ArgumentParser(description="Inspect CSV landmark statistics.")
    ap.add_argument("--csv",
                    help="Single CSV to inspect (overrides train/val).")
    ap.add_argument("--train_csv",
                    default=TRAIN_CSV_DEFAULT,
                    help="Path to train CSV (used if --csv not provided).")
    ap.add_argument("--val_csv",
                    default=VAL_CSV_DEFAULT,
                    help="Path to val CSV (used if --csv not provided).")
    ap.add_argument("--landmark_count",
                    type=int,
                    default=EXPECTED_L_DEFAULT,
                    help="Expected number of landmarks per row.")
    ap.add_argument("--max_show",
                    type=int,
                    default=15,
                    help="Max suspicious rows to print per category.")
    return ap.parse_args()

# ---------------------- Parsers ----------------------

def strict_parse_visibility(v):
    """
    STRICT: only 'True' (case-insensitive) counts as True;
    only 'False' counts as False; everything else treated as False.
    Prevents lenient parsing that might inflate visible counts.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        # Only 1 -> True, 0 -> False; other ints considered True if non-zero
        try:
            return bool(int(v))
        except Exception:
            return False
    if not isinstance(v, str):
        return False
    s = v.strip()
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    return False

def parse_landmarks(raw):
    """
    Parse "x,y; x,y; ..." into a list of (x,y) floats.
    Returns [] if empty or invalid.
    """
    if not isinstance(raw, str):
        return []
    t = raw.strip()
    if t == "" or t.lower() in ("[]","nan","none","null"):
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

def bbox_valid(raw: str) -> bool:
    if not isinstance(raw, str):
        return False
    s = raw.strip()
    if s.lower() in ("","nan","none","null"):
        return False
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        return False
    try:
        x1,y1,x2,y2 = map(float, parts)
    except ValueError:
        return False
    if x2 <= x1 or y2 <= y1:
        return False
    return True

# ---------------------- Core Stats ----------------------

def analyze_csv(path: str, expected_L: int, max_show: int):
    print("="*70)
    print(f"CSV: {path}")
    if not Path(path).exists():
        print("ERROR: File does not exist.")
        return

    df = pd.read_csv(path)
    total_rows = len(df)
    print(f"Total rows: {total_rows}")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        return

    vis_list = [strict_parse_visibility(v) for v in df.eye_visibility]
    bbox_list = [bbox_valid(b) for b in df.eye_bbox_face]
    lm_list = [parse_landmarks(l) for l in df.landmarks_coordinates_inside_eye_bbox]
    lm_counts = [len(lm) for lm in lm_list]
    has_expected_landmarks = [c == expected_L for c in lm_counts]

    vis_counter = Counter(vis_list)
    bbox_counter = Counter(bbox_list)
    expected_lm_counter = Counter(has_expected_landmarks)

    print("\nVisibility (strict) counts:", dict(vis_counter))
    print(f"Valid bbox count: {bbox_counter.get(True,0)} | Invalid/empty: {bbox_counter.get(False,0)}")
    print(f"Rows with exactly {expected_L} landmarks: {expected_lm_counter.get(True,0)}")
    print("Rows with other landmark counts (including 0):", expected_lm_counter.get(False,0))

    # Joint stats
    true_and_6   = sum(1 for v,h in zip(vis_list,has_expected_landmarks) if v and h)
    false_and_6  = sum(1 for v,h in zip(vis_list,has_expected_landmarks) if (not v) and h)
    true_and_bb  = sum(1 for v,b in zip(vis_list,bbox_list) if v and b)
    false_and_bb = sum(1 for v,b in zip(vis_list,bbox_list) if (not v) and b)

    print(f"\nVisible & {expected_L} landmarks: {true_and_6}")
    print(f"Invisible & {expected_L} landmarks: {false_and_6}")
    print(f"Visible & valid bbox: {true_and_bb}")
    print(f"Invisible & valid bbox: {false_and_bb}")

    # Eye side
    side_counter = Counter(df.eye_side)
    side_vis_true  = Counter(df.eye_side[i] for i,v in enumerate(vis_list) if v)
    side_vis_false = Counter(df.eye_side[i] for i,v in enumerate(vis_list) if not v)

    print("\nEye side counts:", dict(side_counter))
    print("Eye side visible:", dict(side_vis_true))
    print("Eye side invisible:", dict(side_vis_false))

    # Suspicious sets
    if false_and_6 > 0:
        print(f"\nSuspicious rows (visibility=False but {expected_L} landmarks present): {false_and_6}")
        shown = 0
        for i,(v,h) in enumerate(zip(vis_list,has_expected_landmarks)):
            if not v and h:
                row = df.iloc[i]
                print(f"[False&6] idx={i} video_id={row.video_id} frame_key={row.frame_key} "
                      f"eye_side={row.eye_side} bbox='{row.eye_bbox_face}' lm='{row.landmarks_coordinates_inside_eye_bbox}'")
                shown += 1
                if shown >= max_show:
                    break
    else:
        print("\nNo rows where visibility=False contain the expected landmark count.")

    if false_and_bb > 0:
        print(f"\nInvisible rows with valid bbox: {false_and_bb}")
        shown = 0
        for i,(v,b) in enumerate(zip(vis_list,bbox_list)):
            if not v and b:
                row = df.iloc[i]
                print(f"[False&BBox] idx={i} video_id={row.video_id} frame_key={row.frame_key} "
                      f"eye_side={row.eye_side} bbox='{row.eye_bbox_face}' landmark_len={lm_counts[i]}")
                shown += 1
                if shown >= max_show:
                    break
    else:
        print("\nNo invisible rows have a valid bbox.")

    # Summary explanation
    print("\n--- Summary Interpretation ---")
    print("1. If 'Invisible & {expected_L} landmarks' is 0, invisible rows truly lack the full landmark set.")
    print("2. If evaluation earlier reported more samples than visible count (e.g., 1636 > 1410), those extra rows are")
    print("   either invisible with valid bbox/landmarks or evaluator did not filter at all.")
    print("3. Use --visible_only at evaluation time to restrict to strictly visible rows.")
    print("4. This script uses STRICT visibility parsing; only exact 'True'/'False' (case-insensitive) are recognized.")
    print("="*70)

def main():
    args = parse_args()

    if args.csv:
        # Single CSV mode
        analyze_csv(args.csv, args.landmark_count, args.max_show)
    else:
        # Train + Val mode
        analyze_csv(args.train_csv, args.landmark_count, args.max_show)
        analyze_csv(args.val_csv,   args.landmark_count, args.max_show)

if __name__ == "__main__":
    main()