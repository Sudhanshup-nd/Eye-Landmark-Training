# from typing import Dict, Any
# from PIL import Image
# import torch
# from torchvision import transforms
# from .utils import debug  # now valid with __init__.py

# class ToTensorContiguous:
#     def __init__(self, verbose: bool = False):
#         self.verbose = verbose

#     def __call__(self, img: Image.Image) -> torch.Tensor:
#         if self.verbose:
#             debug(f"ToTensor input size={img.size}", True)
#         t = transforms.ToTensor()(img).contiguous()
#         if self.verbose:
#             debug(f"ToTensor output shape={tuple(t.shape)}", True)
#         return t

# def build_train_transforms(cfg: Dict[str, Any]):
#     data_cfg = cfg.get("data", {})
#     dbg_cfg = cfg.get("debug", {})
#     verbose = bool(dbg_cfg.get("verbose_transforms", False))

#     tf_list = []
#     jitter_cfg = data_cfg.get("color_jitter", {})
#     brightness = jitter_cfg.get("brightness", 0.0)
#     contrast   = jitter_cfg.get("contrast", 0.0)
#     saturation = jitter_cfg.get("saturation", 0.0)
#     hue        = jitter_cfg.get("hue", 0.0)

#     if any(v > 0 for v in [brightness, contrast, saturation, hue]):
#         tf_list.append(
#             transforms.ColorJitter(
#                 brightness=brightness,
#                 contrast=contrast,
#                 saturation=saturation,
#                 hue=hue
#             )
#         )

#     tf_list.append(ToTensorContiguous(verbose=verbose))

#     if "mean" in data_cfg and "std" in data_cfg:
#         tf_list.append(transforms.Normalize(mean=data_cfg["mean"], std=data_cfg["std"]))

#     return transforms.Compose(tf_list)

# def build_val_transforms(cfg: Dict[str, Any]):
#     data_cfg = cfg.get("data", {})
#     dbg_cfg = cfg.get("debug", {})
#     verbose = bool(dbg_cfg.get("verbose_transforms", False))

#     tf_list = [ToTensorContiguous(verbose=verbose)]
#     if "mean" in data_cfg and "std" in data_cfg:
#         tf_list.append(transforms.Normalize(mean=data_cfg["mean"], std=data_cfg["std"]))
#     return transforms.Compose(tf_list)






from typing import Dict, Any
from PIL import Image
import torch
from torchvision import transforms
from .utils import debug  # now valid with __init__.py

class ToTensorContiguous:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if self.verbose:
            debug(f"ToTensor input size={img.size}", True)
        t = transforms.ToTensor()(img).contiguous()
        if self.verbose:
            debug(f"ToTensor output shape={tuple(t.shape)}", True)
        return t

def build_train_transforms(cfg: Dict[str, Any]):
    """
    Updated to keep photometric transforms here (e.g., ColorJitter),
    while geometric, landmark-aware ops are handled in src/augmentations.py
    within the dataset __getitem__ before ToTensor/Normalize.
    """
    data_cfg = cfg.get("data", {})
    dbg_cfg = cfg.get("debug", {})
    verbose = bool(dbg_cfg.get("verbose_transforms", False))

    tf_list = []

    # Optional photometric augmentation (safe for landmarks)
    jitter_cfg = data_cfg.get("color_jitter", {})
    brightness = float(jitter_cfg.get("brightness", 0.0))
    contrast   = float(jitter_cfg.get("contrast", 0.0))
    saturation = float(jitter_cfg.get("saturation", 0.0))
    hue        = float(jitter_cfg.get("hue", 0.0))

    if any(v > 0.0 for v in [brightness, contrast, saturation, hue]):
        tf_list.append(
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue
            )
        )

    # Tensor + Normalize
    tf_list.append(ToTensorContiguous(verbose=verbose))
    if "mean" in data_cfg and "std" in data_cfg:
        tf_list.append(transforms.Normalize(mean=data_cfg["mean"], std=data_cfg["std"]))

    return transforms.Compose(tf_list)

def build_val_transforms(cfg: Dict[str, Any]):
    """
    Validation transforms: no photometric jitter; just tensor + normalize.
    """
    data_cfg = cfg.get("data", {})
    dbg_cfg = cfg.get("debug", {})
    verbose = bool(dbg_cfg.get("verbose_transforms", False))

    tf_list = [ToTensorContiguous(verbose=verbose)]
    if "mean" in data_cfg and "std" in data_cfg:
        tf_list.append(transforms.Normalize(mean=data_cfg["mean"], std=data_cfg["std"]))
    return transforms.Compose(tf_list)