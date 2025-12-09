#!/usr/bin/env python3
"""
Standalone dataset diagnostic utility.
Usage:
  python scripts/diagnose_dataset.py --csv path/to/train.csv --num 50
Prints:
 - Visibility raw counts
 - Landmark presence counts
 - BBox validity
 - Sample rows with mismatches
"""

import argparse
import pandas as pd
from collections import Counter

def has_landmarks(s: str) -> bool:
    if not isinstance(s, str):
        return False
    st = s.strip().lower()
    if st in ("", "[]", "nan", "none", "null"):
        return False
    return "," in s

def bbox_valid(s: str) -> bool:
    if not isinstance(s, str):
        return False
    parts = s.strip().split(",")
    if len(parts) != 4:
        return False
    try:
        nums = list(map(float, parts))
    except ValueError:
        return False
    return not (nums[0] == 0 and nums[1] == 0 and nums[2] == 0 and nums[3] == 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--num", type=int, default=50)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")

    raw_vis = df.eye_visibility.astype(str).str.strip().str.lower()
    print("Raw visibility counts:", raw_vis.value_counts(dropna=False).to_dict())

    lm_presence = raw_vis.copy()
    lm_presence = df.landmarks_coordinates_inside_eye_bbox.astype(str).apply(has_landmarks)
    print("Landmark presence counts:", Counter(lm_presence))

    bbox_presence = df.eye_bbox_face.astype(str).apply(bbox_valid)
    print("BBox valid counts:", Counter(bbox_presence))

    mismatches = []
    for i in range(min(args.num, len(df))):
        v = raw_vis.iloc[i]
        lmp = lm_presence.iloc[i]
        if (v in ("true","1") and not lmp) or (lmp and v in ("false","0","", "nan", "none", "null")):
            mismatches.append(i)

    if mismatches:
        print(f"MISMATCH indices (first 20): {mismatches[:20]}")
        for mi in mismatches[:5]:
            row = df.iloc[mi]
            print(f"Row {mi}: visibility={row.eye_visibility} landmarks_str='{row.landmarks_coordinates_inside_eye_bbox}' bbox='{row.eye_bbox_face}'")
    else:
        print("No visibility/landmark mismatches in first sampled rows.")

if __name__ == "__main__":
    main()