import os
import re
from typing import Any, Dict, List, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

from .augmentations import apply_augmentations  # <-- integrate augmentations

def _dbg(enabled: bool, msg: str):
    if enabled:
        print(f"[DATASET-DEBUG] {msg}")

class EyeDataset(Dataset):
    def __init__(self, csv_path: str, cfg: Dict[str, Any], transform=None, is_train: bool = True):
        self.csv_path = csv_path
        self.cfg = cfg
        self.transform = transform
        self.is_train = is_train
        self.df = pd.read_csv(csv_path)

        required = [
            "video_id","frame_key","eye_side","eye_visibility",
            "path_to_dataset","eye_bbox_face","landmarks_coordinates_inside_eye_bbox"
        ]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV {csv_path} missing columns: {missing}")

        dcfg = cfg.get("data", {})
        self.num_landmarks = int(dcfg.get("num_landmarks", 6))
        self.normalize_landmarks = bool(dcfg.get("normalize_landmarks", True))
        self.use_local_coords = bool(dcfg.get("landmarks_local_coords", True))
        self.infer_visibility = bool(dcfg.get("infer_visibility_from_landmarks", False))
        self.keep_landmarks_if_invisible = True  # override per your spec

        self.image_size = int(dcfg.get("image_size", 128))
        self.debug_enabled = bool(self.cfg.get("debug", {}).get("enabled", False))
        self.length = len(self.df)

        self.valid_indices = [
            i for i in range(self.length)
            if self._has_landmarks(str(self.df.iloc[i].landmarks_coordinates_inside_eye_bbox))
        ]
        _dbg(self.debug_enabled, f"Filtered: {len(self.valid_indices)} valid landmark rows out of {self.length}")
        self._initial_diagnostics()

    def _initial_diagnostics(self):
        dbg = self.debug_enabled
        _dbg(dbg, f"Loaded {self.length} raw rows; using {len(self.valid_indices)} with landmarks.")
        raw_vis = self.df.eye_visibility.astype(str).str.strip().str.lower()
        _dbg(dbg, f"Raw visibility counts: {raw_vis.value_counts(dropna=False).to_dict()}")   #counts even if eye_visibility is NaN

    def _has_landmarks(self, s: str) -> bool:
        s = s.strip().lower()
        if s in ("", "[]", "nan", "none", "null"):
            return False
        return "," in s

    def __len__(self):
        return len(self.valid_indices)

    def _sanitize_visibility_string(self, raw: Any) -> str:
        s = str(raw).strip().lower()
        if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
            s = s[1:-1].strip().lower()
        m = re.search(r"[a-z0-9]+", s)
        return m.group(0) if m else ""

    def _parse_visibility_raw(self, raw: Any) -> int:
        if raw is None:
            return 0
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            try:
                if isinstance(raw, float) and torch.isnan(torch.tensor(raw)):
                    return 0
            except Exception:
                pass
            return 1 if int(raw) != 0 else 0
        token = self._sanitize_visibility_string(raw)
        if token in ("true", "1", "yes", "y", "visible"):
            return 1
        if token in ("false", "0", "no", "n", "invisible"):
            return 0
        try:
            return 1 if int(token) != 0 else 0
        except Exception:
            return 0

    def _parse_bbox(self, raw: Any) -> Tuple[int,int,int,int]:
        if not isinstance(raw, str):
            return (0,0,0,0)
        r = raw.strip().lower()
        if r in ("","nan","none","null"):
            return (0,0,0,0)
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 4:
            return (0,0,0,0)
        try:
            nums = list(map(int, parts))
        except ValueError:
            try:
                nums = list(map(float, parts))    #converts to float first
                nums = list(map(int, nums))      #then convert to int
            except Exception:
                return (0,0,0,0)
        x1,y1,x2,y2 = nums
        return (x1,y1,x2,y2)

    def _parse_landmarks(self, raw: Any) -> List[Tuple[float,float]]:
        if not isinstance(raw, str): return []
        t = raw.strip()
        if t == "" or t.lower() in ("[]","nan","none","null"): return []
        pts = []
        for token in t.split(";"):
            tok = token.strip()
            if not tok:
                continue
            seg = tok.split(",")
            if len(seg) != 2:
                continue
            try:
                x = float(seg[0]); y = float(seg[1])
                pts.append((x,y))
            except ValueError:
                continue
        return pts

    def _crop_eye(self, img: Image.Image, bbox: Tuple[int,int,int,int]) -> Image.Image:
        w_face, h_face = img.size
        x1,y1,x2,y2 = bbox
        x1 = max(0, min(x1, w_face-1))
        x2 = max(0, min(x2, w_face-1))
        y1 = max(0, min(y1, h_face-1))
        y2 = max(0, min(y2, h_face-1))
        if x2 < x1: x1,x2 = x2,x1
        if y2 < y1: y1,y2 = y2,y1

        crop = img.crop((x1, y1, x2+1, y2+1))
        cw, ch = crop.size
        if cw != ch:
            side = max(cw, ch)
            pad_w = side - cw
            pad_h = side - ch
            new_img = Image.new("RGB", (side, side), (0,0,0))
            new_img.paste(crop, (pad_w//2, pad_h//2))
            crop = new_img
        if crop.size != (self.image_size, self.image_size):
            crop = crop.resize((self.image_size, self.image_size), Image.BILINEAR)
        return crop

    def _landmark_tensor(self, pts: List[Tuple[float,float]], bbox: Tuple[int,int,int,int]):
        n = self.num_landmarks
        lm = torch.zeros((n,2), dtype=torch.float32)
        mask = torch.zeros((n,), dtype=torch.float32)
        if len(pts) == 0:
            return lm, mask
        x1,y1,x2,y2 = bbox
        bw = max(1.0, float(x2 - x1 + 1))
        bh = max(1.0, float(y2 - y1 + 1))
        for i,(x,y) in enumerate(pts[:n]):
            lx = x - x1 if self.use_local_coords else x
            ly = y - y1 if self.use_local_coords else y
            if self.normalize_landmarks:
                lx /= bw; ly /= bh
            lm[i,0] = lx; lm[i,1] = ly
            mask[i] = 1.0
        return lm, mask

    def __getitem__(self, index: int):
        real_idx = self.valid_indices[index]
        row = self.df.iloc[real_idx]

        raw_vis_value = row.eye_visibility
        vis_raw = self._parse_visibility_raw(raw_vis_value)
        bbox = self._parse_bbox(row.eye_bbox_face)
        pts  = self._parse_landmarks(row.landmarks_coordinates_inside_eye_bbox)

        vis_final = vis_raw
        if self.infer_visibility and vis_raw == 0 and len(pts) > 0:
            vis_final = 1

        img_path = row.path_to_dataset
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image missing: {img_path}")
        face_img = Image.open(img_path).convert("RGB")
        eye_img = self._crop_eye(face_img, bbox)

        # Build normalized local landmark coordinates BEFORE augmentation
        x1,y1,x2,y2 = bbox
        bw = max(1.0, float(x2 - x1 + 1))
        bh = max(1.0, float(y2 - y1 + 1))
        landmarks_norm: List[Tuple[float,float]] = []
        for (x,y) in pts[:self.num_landmarks]:
            lx = (x - x1) / bw
            ly = (y - y1) / bh
            landmarks_norm.append((lx, ly))

        # Apply landmark-aware augmentations on PIL eye crop and normalized landmarks
        eye_img, landmarks_norm = apply_augmentations(eye_img, landmarks_norm, self.cfg, self.is_train)

        # Convert augmented normalized landmarks to tensor
        lm = torch.zeros((self.num_landmarks,2), dtype=torch.float32)
        mask = torch.zeros((self.num_landmarks,), dtype=torch.float32)
        for i, (lx, ly) in enumerate(landmarks_norm[:self.num_landmarks]):
            lm[i,0] = float(lx)
            lm[i,1] = float(ly)
            mask[i] = 1.0

        if self.transform:
            eye_img = self.transform(eye_img)
        else:
            import torchvision.transforms as T
            eye_img = T.ToTensor()(eye_img).contiguous()

        sample = {
            "image": eye_img,
            "visibility": torch.tensor([vis_final], dtype=torch.float32),
            "landmarks": lm,
            "mask": mask,
            "bbox": torch.tensor(bbox, dtype=torch.float32),
            "video_id": row.video_id,
            "frame_key": row.frame_key,
            "eye_side": row.eye_side,
            "raw_visibility": torch.tensor([vis_raw], dtype=torch.float32),
            "raw_landmarks_count": torch.tensor([len(pts)], dtype=torch.float32),
            "raw_visibility_value_str": str(raw_vis_value)
        }

        if self.debug_enabled and index < 5:
            _dbg(True, f"SAMPLE idx={index} frame={row.frame_key} eye={row.eye_side} "
                       f"vis_raw={vis_raw} vis_final={vis_final} "
                       f"lm_count={len(pts)} mask_sum={mask.sum().item()} bbox={bbox}")

        return sample