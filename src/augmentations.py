"""
Placeholder for future landmark-aware augmentations.
Currently unused (geometric transforms disabled).
You can implement functions here that accept both image and landmarks,
adjust landmark coordinates accordingly (e.g., horizontal flip, rotation).
"""

# Example skeleton:
# def horizontal_flip(img, landmarks):
#     w = img.width
#     flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
#     new_landmarks = [(w - 1 - x, y) for (x,y) in landmarks]
#     return flipped_img, new_landmarks




"""
Landmark-aware augmentations for eye crops.

Augmentations operate in CROPPED image space (square eye crop before normalization),
so landmarks are expected in local crop coordinates:
- If cfg.data.landmarks_local_coords == True and cfg.data.normalize_landmarks == True,
  we will apply geometric transforms to the PIL image and correspondingly update landmarks
  in normalized [0,1] coordinates.

Supported ops (controlled by cfg['data']['augment']):
- random_horizontal_flip: probability in [0,1]
- random_vertical_flip:   probability in [0,1]
- random_rotation_deg:    max absolute rotation degrees (uniform in [-deg, deg])
- occlusion_p:            probability to add a random rectangular occluder
- blur_p:                 probability to apply slight Gaussian blur

Note:
- Rotation is around image center.
- Landmarks are updated to remain consistent after flips/rotation.
- Occlusion and blur do not change landmarks.
"""

from typing import List, Tuple, Dict, Any
import random
import math
from PIL import Image, ImageFilter, ImageDraw

def _rand_bool(p: float) -> bool:
    return p > 0 and random.random() < p

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _rotate_point_norm(x: float, y: float, angle_rad: float) -> Tuple[float, float]:
    """
    Rotate normalized point (x,y) around center (0.5,0.5) by angle_rad.
    """
    cx, cy = 0.5, 0.5
    dx = x - cx
    dy = y - cy
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rx = dx * cos_a - dy * sin_a
    ry = dx * sin_a + dy * cos_a
    return (_clip01(cx + rx), _clip01(cy + ry))

def _horizontal_flip_norm(x: float, y: float) -> Tuple[float, float]:
    return (_clip01(1.0 - x), y)

def _vertical_flip_norm(x: float, y: float) -> Tuple[float, float]:
    return (x, _clip01(1.0 - y))

def _apply_random_occlusion(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w < 4 or h < 4:
        return img
    # Random rectangle occluder (black)
    occ_w = random.randint(max(2, w // 8), max(3, w // 3))
    occ_h = random.randint(max(2, h // 8), max(3, h // 3))
    x1 = random.randint(0, max(0, w - occ_w))
    y1 = random.randint(0, max(0, h - occ_h))
    x2 = x1 + occ_w
    y2 = y1 + occ_h
    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
    return img

def _apply_random_blur(img: Image.Image) -> Image.Image:
    # Slight blur
    radius = random.uniform(0.5, 1.5)
    return img.filter(ImageFilter.GaussianBlur(radius))

def apply_augmentations(
    img: Image.Image,
    landmarks_norm: List[Tuple[float, float]],
    cfg: Dict[str, Any],
    is_train: bool
) -> Tuple[Image.Image, List[Tuple[float, float]]]:
    """
    Apply landmark-aware augmentations to a cropped, square eye image (PIL),
    and adjust normalized landmarks accordingly.

    landmarks_norm: list of (x,y) in [0,1] relative to the crop.
    Returns augmented image and updated landmarks.
    """
    if not is_train:
        return img, landmarks_norm

    aug = cfg.get("data", {}).get("augment", {}) or {}

    # Copy landmarks to modify
    lmk = [(float(x), float(y)) for (x, y) in landmarks_norm]

    # Geometric transforms
    # Horizontal flip
    if _rand_bool(float(aug.get("random_horizontal_flip", 0.0))):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        lmk = [_horizontal_flip_norm(x, y) for (x, y) in lmk]

    # Vertical flip
    if _rand_bool(float(aug.get("random_vertical_flip", 0.0))):
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        lmk = [_vertical_flip_norm(x, y) for (x, y) in lmk]

    # Rotation (uniform in [-deg, deg])
    max_deg = float(aug.get("random_rotation_deg", 0.0))
    if max_deg > 0.0:
        angle = random.uniform(-max_deg, max_deg)
        angle_rad = math.radians(angle)
        # PIL rotates around center, expand=False keeps same size
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
        lmk = [_rotate_point_norm(x, y, angle_rad) for (x, y) in lmk]

    # Photometric transforms (no landmark changes)
    if _rand_bool(float(aug.get("occlusion_p", 0.0))):
        img = _apply_random_occlusion(img)

    if _rand_bool(float(aug.get("blur_p", 0.0))):
        img = _apply_random_blur(img)

    return img, lmk