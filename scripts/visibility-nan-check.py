#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

INPUT = Path("/inwdata2a/sudhanshu/eye_multitask_training/data/val-v2.csv")
OUTPUT_CLEAN = Path("/inwdata2a/sudhanshu/eye_multitask_training/data/val-v2.clean.csv")
OUTPUT_NAN_ROWS = Path("/inwdata2a/sudhanshu/eye_multitask_training/data/val-v2.nan_visibility_rows.csv")

REPLACE_NAN_WITH = "false"    # or "0" if you prefer numeric

def main():
    if not INPUT.exists():
        print(f"File not found: {INPUT}")
        return

    df = pd.read_csv(INPUT)
    nan_mask = df["eye_visibility"].isna()
    nan_rows = df[nan_mask]

    print(f"Total rows: {len(df)}")
    print(f"NaN eye_visibility rows: {nan_rows.shape[0]}")

    if nan_rows.empty:
        print("No NaN visibility rows.")
        return

    print("\n--- Rows with NaN eye_visibility ---")
    # Print full rows (could be wide; adjust as needed)
    print(nan_rows.to_string(index=True))

    print("\n--- Individual Row Dicts ---")
    for idx, row in nan_rows.iterrows():
        print(f"Row index {idx}:")
        print(row.to_dict())
        print("-" * 60)

    # Save the NaN rows separately
    nan_rows.to_csv(OUTPUT_NAN_ROWS, index=False)
    print(f"Saved NaN rows to: {OUTPUT_NAN_ROWS}")

    # Create a cleaned version
    df.loc[nan_mask, "eye_visibility"] = REPLACE_NAN_WITH
    df.to_csv(OUTPUT_CLEAN, index=False)
    print(f"Cleaned CSV written to: {OUTPUT_CLEAN} (NaNs replaced with '{REPLACE_NAN_WITH}')")

if __name__ == "__main__":
    main()