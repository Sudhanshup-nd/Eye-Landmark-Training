# #!/usr/bin/env python3
# """
# Generate per-eye rows from a CVAT XML dump containing left/right eye landmarks and visibility tags.

# Output CSV columns:
# video_id,frame_key,eye_side,eye_visibility,path_to_dataset,eye_bbox_face,landmarks_coordinates_inside_eye_bbox

# Assumptions:
# - <image name="Eyelid-landmarks-labelling/<video_id>/<frame_file>.jpg">
# - Landmarks labels: 'left-eye-landmarks', 'right-eye-landmarks'
# - Visibility tags: 'left-eye-visibility', 'right-eye-visibility' with attribute name="is-visible"
# - Bounding box is computed from the 6 landmark points plus padding.
# - landmarks_coordinates_inside_eye_bbox are taken directly from XML (absolute coords), not normalized.

# You can edit the configuration variables below instead of using CLI arguments.
# """

# import csv
# import os
# import xml.etree.ElementTree as ET
# from typing import List, Tuple, Dict

# # -----------------------------------------------------------------------------
# # Configuration (EDIT THESE AS NEEDED)
# # -----------------------------------------------------------------------------
# XML_PATH = "/inwdata2a/sudhanshu/data-and-labels/annotations 3.xml"  # Path to CVAT XML dump
# ROOT_DIR = "/inwdata2a/sudhanshu/data-and-labels/labelling-data"  # Root dir containing video_id folders
# OUT_CSV = "/absolute/path/for/output/eye_landmarks.csv"  # Output CSV path
# PADDING_RATIO = 0.10          # Fraction of bbox width/height to expand
# SKIP_MISSING_EYE = False       # If True, skip eyes with no landmarks
# # -----------------------------------------------------------------------------

# LEFT_LANDMARK_LABEL = "left-eye-landmarks"
# RIGHT_LANDMARK_LABEL = "right-eye-landmarks"
# LEFT_VIS_LABEL = "left-eye-visibility"
# RIGHT_VIS_LABEL = "right-eye-visibility"


# def parse_points(points_str: str) -> List[Tuple[float, float]]:
#     """Parse points attribute: 'x1,y1;x2,y2;...' into list of (x,y)."""
#     pts = []
#     for pair in points_str.strip().split(";"):
#         pair = pair.strip()
#         if not pair:
#             continue
#         x_str, y_str = pair.split(",")
#         pts.append((float(x_str), float(y_str)))
#     return pts


# def bbox_with_padding(
#     pts: List[Tuple[float, float]],
#     img_w: int,
#     img_h: int,
#     padding_ratio: float
# ) -> Tuple[int, int, int, int]:
#     """Compute padded bbox (xmin, ymin, xmax, ymax) clamped to image dimensions."""
#     xs = [p[0] for p in pts]
#     ys = [p[1] for p in pts]
#     xmin = min(xs)
#     xmax = max(xs)
#     ymin = min(ys)
#     ymax = max(ys)
#     width = xmax - xmin
#     height = ymax - ymin

#     pad_x = width * padding_ratio
#     pad_y = height * padding_ratio

#     xmin_p = max(0, int(round(xmin - pad_x)))
#     ymin_p = max(0, int(round(ymin - pad_y)))
#     xmax_p = min(img_w - 1, int(round(xmax + pad_x)))
#     ymax_p = min(img_h - 1, int(round(ymax + pad_y)))

#     return xmin_p, ymin_p, xmax_p, ymax_p


# def extract_video_id_and_frame_key(image_name: str) -> Tuple[str, str]:
#     """
#     From name like: 'Eyelid-landmarks-labelling/<video_id>/frame_0030_face_0.jpg'
#     Return (video_id, frame_key_without_extension).
#     """
#     parts = image_name.strip().split("/")
#     if len(parts) < 3:
#         return ("UNKNOWN_VIDEO", os.path.splitext(os.path.basename(image_name))[0])
#     video_id = parts[-2]
#     frame_key = os.path.splitext(parts[-1])[0]
#     return video_id, frame_key


# def build_image_path(root_dir: str, video_id: str, frame_file: str) -> str:
#     """Construct filesystem path: root_dir/video_id/frame_file."""
#     return os.path.join(root_dir, video_id, frame_file)


# def collect_visibility_tags(image_elem: ET.Element) -> Dict[str, str]:
#     """Map visibility label -> 'true'/'false'."""
#     vis = {}
#     for tag in image_elem.findall("tag"):
#         label = tag.get("label", "")
#         if label in (LEFT_VIS_LABEL, RIGHT_VIS_LABEL):
#             attr = tag.find("attribute[@name='is-visible']")
#             if attr is not None and attr.text is not None:
#                 vis[label] = attr.text.strip()
#             else:
#                 vis[label] = ""
#     return vis


# def generate_rows(xml_path: str) -> List[Dict[str, str]]:
#     tree = ET.parse(xml_path)
#     root = tree.getroot()

#     rows = []
#     for image in root.findall("image"):
#         image_name = image.get("name", "")
#         img_w = int(image.get("width", "0"))
#         img_h = int(image.get("height", "0"))

#         video_id, frame_key = extract_video_id_and_frame_key(image_name)
#         frame_file = os.path.basename(image_name)
#         path_to_dataset = build_image_path(ROOT_DIR, video_id, frame_file)

#         # Gather landmark points
#         landmarks_map = {}
#         raw_points_str_map = {}
#         for pe in image.findall("points"):
#             label = pe.get("label", "")
#             pts_str = pe.get("points", "")
#             if not pts_str:
#                 continue
#             try:
#                 pts = parse_points(pts_str)
#             except Exception:
#                 continue
#             landmarks_map[label] = pts
#             raw_points_str_map[label] = pts_str

#         visibility_map = collect_visibility_tags(image)

#         for eye_label, side in [(LEFT_LANDMARK_LABEL, "left"), (RIGHT_LANDMARK_LABEL, "right")]:
#             pts = landmarks_map.get(eye_label)
#             if pts is None:
#                 if SKIP_MISSING_EYE:
#                     continue
#                 rows.append({
#                     "video_id": video_id,
#                     "frame_key": frame_key,
#                     "eye_side": side,
#                     "eye_visibility": visibility_map.get(
#                         LEFT_VIS_LABEL if side == "left" else RIGHT_VIS_LABEL, ""
#                     ),
#                     "path_to_dataset": path_to_dataset,
#                     "eye_bbox_face": "",
#                     "landmarks_coordinates_inside_eye_bbox": ""
#                 })
#                 continue

#             xmin, ymin, xmax, ymax = bbox_with_padding(pts, img_w, img_h, PADDING_RATIO)
#             bbox_str = f"{xmin},{ymin},{xmax},{ymax}"
#             visibility = visibility_map.get(
#                 LEFT_VIS_LABEL if side == "left" else RIGHT_VIS_LABEL, ""
#             )

#             rows.append({
#                 "video_id": video_id,
#                 "frame_key": frame_key,
#                 "eye_side": side,
#                 "eye_visibility": visibility,
#                 "path_to_dataset": path_to_dataset,
#                 "eye_bbox_face": bbox_str,
#                 "landmarks_coordinates_inside_eye_bbox": raw_points_str_map.get(eye_label, "")
#             })
#     return rows


# def write_csv(rows: List[Dict[str, str]], out_csv: str):
#     fieldnames = [
#         "video_id",
#         "frame_key",
#         "eye_side",
#         "eye_visibility",
#         "path_to_dataset",
#         "eye_bbox_face",
#         "landmarks_coordinates_inside_eye_bbox"
#     ]
#     os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
#     with open(out_csv, "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         for r in rows:
#             writer.writerow(r)


# def main():
#     rows = generate_rows(XML_PATH)
#     write_csv(rows, OUT_CSV)
#     print(f"Saved {len(rows)} rows (expected ~2 x frames) to {OUT_CSV}")


# if __name__ == "__main__":
#     main()










#!/usr/bin/env python3
"""
Same as previous CSV generator, but makes EYE BBOXES SQUARE.

Only change from your working script:
- The bbox computation now expands to a square whose side = max(width, height),
  then applies the padding ratio uniformly to that side, and clamps to image bounds.

All other behavior (columns, logging, etc.) is unchanged.

If you want to revert to rectangular, just replace the square_bbox_with_padding()
call with your original bbox_with_padding().

Configuration section remains the same.
"""

import csv
import os
import sys
import traceback
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict

# =============================================================================
# CONFIGURATION (EDIT THESE PATHS / FLAGS)
# =============================================================================
XML_PATH = "/inwdata2a/sudhanshu/data-and-labels/annotations 3.xml"   # Path to CVAT export XML
ROOT_DIR = "/inwdata2a/sudhanshu/data-and-labels/labelling-data"      # Root dir containing video_id subfolders
OUT_CSV = "/inwdata2a/sudhanshu/eye_multitask_training/outputs/eye_landmarks.csv"  # Destination CSV
PADDING_RATIO = 0.5                 # Fractional padding added to the SQUARE side length
SKIP_MISSING_EYE = False            # If True, skip row entirely when an eye lacks landmarks
EXPECTED_POINTS_PER_EYE = 6         # For validation; just warns if mismatch
DEBUG = True                        # Toggle verbose debug
SHOW_SAMPLE_ROWS = 5                # Number of sample rows to print after processing
# =============================================================================

LEFT_LANDMARK_LABEL = "left-eye-landmarks"
RIGHT_LANDMARK_LABEL = "right-eye-landmarks"
LEFT_VIS_LABEL = "left-eye-visibility"
RIGHT_VIS_LABEL = "right-eye-visibility"


def debug(msg: str):
    if DEBUG:
        print(f"[DEBUG] {msg}")


def parse_points(points_str: str) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    if not points_str:
        return pts
    for raw_pair in points_str.strip().split(";"):
        pair = raw_pair.strip()
        if not pair:
            continue
        try:
            x_str, y_str = pair.split(",")
            pts.append((float(x_str), float(y_str)))
        except Exception as e:
            debug(f"Failed parsing point '{pair}' in '{points_str}': {e}")
    return pts


def square_bbox_with_padding(
    pts: List[Tuple[float, float]],
    img_w: int,
    img_h: int,
    padding_ratio: float
) -> Tuple[int, int, int, int]:
    """
    Create a square bbox around the landmarks with uniform padding.

    Steps:
      1. Get min/max x,y.
      2. side = max(width, height).
      3. padded_side = side * (1 + 2*padding_ratio)  (padding on both sides)
      4. Center square around the original bbox center.
      5. Clamp to image boundaries.
    """
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)

    width = xmax - xmin
    height = ymax - ymin
    side = max(width, height)
    # Ensure non-zero
    if side == 0:
        side = 1.0

    # Apply padding (padding_ratio is fraction of original side per side)
    padded_side = side * (1 + 2 * padding_ratio)

    # Center coordinates
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0

    half = padded_side / 2.0
    sq_xmin = int(round(cx - half))
    sq_xmax = int(round(cx + half))
    sq_ymin = int(round(cy - half))
    sq_ymax = int(round(cy + half))

    # Clamp to image boundaries
    # Horizontal shift if out of bounds
    if sq_xmin < 0:
        shift = -sq_xmin
        sq_xmin += shift
        sq_xmax += shift
    if sq_xmax >= img_w:
        shift = sq_xmax - (img_w - 1)
        sq_xmin -= shift
        sq_xmax -= shift
    # Vertical shift
    if sq_ymin < 0:
        shift = -sq_ymin
        sq_ymin += shift
        sq_ymax += shift
    if sq_ymax >= img_h:
        shift = sq_ymax - (img_h - 1)
        sq_ymin -= shift
        sq_ymax -= shift

    # Final clamp
    sq_xmin = max(0, sq_xmin)
    sq_ymin = max(0, sq_ymin)
    sq_xmax = min(img_w - 1, sq_xmax)
    sq_ymax = min(img_h - 1, sq_ymax)

    # Guarantee ordering
    if sq_xmin > sq_xmax:
        sq_xmin, sq_xmax = sq_xmax, sq_xmin
    if sq_ymin > sq_ymax:
        sq_ymin, sq_ymax = sq_ymax, sq_ymin

    return sq_xmin, sq_ymin, sq_xmax, sq_ymax


def extract_video_id_and_frame_key(image_name: str) -> Tuple[str, str]:
    parts = image_name.strip().split("/")
    if len(parts) < 3:
        debug(f"Unexpected image name format (using fallback): {image_name}")
        return ("UNKNOWN_VIDEO", os.path.splitext(os.path.basename(image_name))[0])
    video_id = parts[-2]
    frame_key = os.path.splitext(parts[-1])[0]
    return video_id, frame_key


def build_image_path(root_dir: str, video_id: str, frame_file: str) -> str:
    return os.path.join(root_dir, video_id, frame_file)


def collect_visibility_tags(image_elem: ET.Element) -> Dict[str, str]:
    vis: Dict[str, str] = {}
    for tag in image_elem.findall("tag"):
        label = tag.get("label", "")
        if label in (LEFT_VIS_LABEL, RIGHT_VIS_LABEL):
            attr = tag.find("attribute[@name='is-visible']")
            vis[label] = (attr.text.strip() if (attr is not None and attr.text) else "")
    return vis


def validate_point_count(side: str, pts: List[Tuple[float, float]]):
    if EXPECTED_POINTS_PER_EYE is not None and len(pts) != EXPECTED_POINTS_PER_EYE:
        debug(f"WARNING: {side} eye landmark count {len(pts)} != expected {EXPECTED_POINTS_PER_EYE}")


def generate_rows(xml_path: str) -> List[Dict[str, str]]:
    debug(f"Starting XML parse: {xml_path}")
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    try:
        tree = ET.parse(xml_path)
    except Exception:
        debug("XML parsing failed; FULL TRACEBACK:")
        traceback.print_exc()
        raise

    root = tree.getroot()
    images = root.findall("image")
    debug(f"Discovered {len(images)} <image> elements.")

    rows: List[Dict[str, str]] = []
    total_images = 0
    missing_left = 0
    missing_right = 0

    for image in images:
        total_images += 1
        image_name = image.get("name", "")
        if not image_name:
            debug("Skipping <image> without 'name' attribute.")
            continue

        try:
            img_w = int(image.get("width", "0"))
            img_h = int(image.get("height", "0"))
        except ValueError:
            debug(f"Invalid width/height for image '{image_name}' - skipping.")
            continue

        video_id, frame_key = extract_video_id_and_frame_key(image_name)
        frame_file = os.path.basename(image_name)
        path_to_dataset = build_image_path(ROOT_DIR, video_id, frame_file)

        if not os.path.isfile(path_to_dataset):
            debug(f"NOTE: Image path does not exist: {path_to_dataset}")

        landmarks_map: Dict[str, List[Tuple[float, float]]] = {}
        raw_points_str_map: Dict[str, str] = {}

        for pe in image.findall("points"):
            label = pe.get("label", "")
            pts_str = pe.get("points", "")
            if not label or not pts_str:
                continue
            pts = parse_points(pts_str)
            if not pts:
                debug(f"No valid points for label '{label}' in '{image_name}'")
                continue
            landmarks_map[label] = pts
            raw_points_str_map[label] = pts_str

        visibility_map = collect_visibility_tags(image)

        for label, side in [(LEFT_LANDMARK_LABEL, "left"), (RIGHT_LANDMARK_LABEL, "right")]:
            pts = landmarks_map.get(label)
            if pts is None:
                if side == "left":
                    missing_left += 1
                else:
                    missing_right += 1

                if SKIP_MISSING_EYE:
                    debug(f"Skipping {side} eye for frame '{frame_key}' (no landmarks).")
                    continue

                row = {
                    "video_id": video_id,
                    "frame_key": frame_key,
                    "eye_side": side,
                    "eye_visibility": visibility_map.get(
                        LEFT_VIS_LABEL if side == "left" else RIGHT_VIS_LABEL, ""
                    ),
                    "path_to_dataset": path_to_dataset,
                    "eye_bbox_face": "",
                    "landmarks_coordinates_inside_eye_bbox": ""
                }
                rows.append(row)
                debug(f"Appended placeholder row for missing {side} eye: frame_key={frame_key}")
                continue

            validate_point_count(side, pts)

            try:
                xmin, ymin, xmax, ymax = square_bbox_with_padding(pts, img_w, img_h, PADDING_RATIO)
                bbox_str = f"{xmin},{ymin},{xmax},{ymax}"
            except Exception as e:
                debug(f"ERROR building square bbox for {side} eye in '{image_name}': {e}")
                bbox_str = ""

            visibility = visibility_map.get(
                LEFT_VIS_LABEL if side == "left" else RIGHT_VIS_LABEL, ""
            )

            row = {
                "video_id": video_id,
                "frame_key": frame_key,
                "eye_side": side,
                "eye_visibility": visibility,
                "path_to_dataset": path_to_dataset,
                "eye_bbox_face": bbox_str,
                "landmarks_coordinates_inside_eye_bbox": raw_points_str_map.get(label, "")
            }
            rows.append(row)
            debug(f"Appended {side} eye row (SQUARE): frame_key={frame_key}, bbox={bbox_str}, visibility={visibility}")

    debug("----- SUMMARY -----")
    debug(f"Total <image> elements processed: {total_images}")
    debug(f"Rows generated: {len(rows)} (aim: 2 per image minus skipped/missing eyes)")
    debug(f"Missing left-eye landmarks:  {missing_left}")
    debug(f"Missing right-eye landmarks: {missing_right}")
    debug("-------------------")

    return rows


def write_csv(rows: List[Dict[str, str]], out_csv: str):
    fieldnames = [
        "video_id",
        "frame_key",
        "eye_side",
        "eye_visibility",
        "path_to_dataset",
        "eye_bbox_face",
        "landmarks_coordinates_inside_eye_bbox"
    ]

    out_dir = os.path.dirname(os.path.abspath(out_csv))
    debug(f"Ensuring output directory exists: {out_dir}")

    try:
        os.makedirs(out_dir, exist_ok=True)
    except PermissionError:
        debug(f"Permission denied creating directory '{out_dir}'.")
        raise
    except Exception as e:
        debug(f"Unexpected directory creation error for '{out_dir}': {e}")
        raise

    debug(f"Writing CSV to: {out_csv}")
    try:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
    except PermissionError:
        debug(f"Permission denied writing file '{out_csv}'.")
        raise
    except Exception as e:
        debug(f"Unexpected error writing CSV '{out_csv}': {e}")
        raise

    debug("CSV write complete.")


def main():
    print("=== Eye Landmark CSV Generation (SQUARE BBOX) ===")
    print(f"XML_PATH:          {XML_PATH}")
    print(f"ROOT_DIR:          {ROOT_DIR}")
    print(f"OUT_CSV:           {OUT_CSV}")
    print(f"PADDING_RATIO:     {PADDING_RATIO}")
    print(f"SKIP_MISSING_EYE:  {SKIP_MISSING_EYE}")
    print(f"EXPECTED_POINTS:   {EXPECTED_POINTS_PER_EYE}")
    print(f"DEBUG:             {DEBUG}")

    try:
        rows = generate_rows(XML_PATH)
    except Exception as e:
        print("FATAL: Error during row generation.")
        traceback.print_exc()
        sys.exit(1)

    if rows:
        print(f"\nSample {min(SHOW_SAMPLE_ROWS, len(rows))} rows:")
        for sample in rows[:SHOW_SAMPLE_ROWS]:
            print(sample)
    else:
        print("No rows generated (check XML, labels, paths).")

    try:
        write_csv(rows, OUT_CSV)
    except Exception as e:
        print("FATAL: Error writing CSV.")
        traceback.print_exc()
        sys.exit(2)

    print(f"\nSUCCESS: Wrote {len(rows)} rows to {OUT_CSV}")
    print("Each eye bbox is now square (with padding).")


if __name__ == "__main__":
    main()