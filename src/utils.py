import os
import yaml
import random
import torch
import numpy as np
import pickle
import datetime
import platform
from typing import Any, Dict, Optional

def timestamp():
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def debug(msg: str, enabled: bool = True):
    if enabled:
        print(f"[DEBUG {timestamp()}] {msg}")

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if "debug" not in cfg:
        cfg["debug"] = {}
    return cfg

def seed_everything(seed: Any, dbg: bool = False):
    seed_int = int(seed)
    random.seed(seed_int)
    np.random.seed(seed_int)
    torch.manual_seed(seed_int)
    torch.cuda.manual_seed_all(seed_int)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    debug(f"Set seeds to {seed_int}", dbg)

def ensure_dir(path: str, dbg: bool = False):
    try:
        os.makedirs(path, exist_ok=True)
        debug(f"Ensured directory: {path}", dbg)
    except PermissionError as e:
        raise PermissionError(f"Cannot create directory '{path}'. {e}")

def save_checkpoint(state: dict, path: str, dbg: bool = False):
    torch.save(state, path)
    debug(f"Saved checkpoint: {path}", dbg)

def save_weights_only(model_state: dict, path: str, dbg: bool = False):
    torch.save(model_state, path)
    debug(f"Saved weights-only file: {path}", dbg)

def load_checkpoint(path: str, map_location: Optional[str] = None,
                    allow_unsafe_fallback: bool = True, dbg: bool = False):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        ckpt = torch.load(path, map_location=map_location)
        debug(f"Loaded checkpoint (safe): {path}", dbg)
        return ckpt
    except pickle.UnpicklingError:
        if allow_unsafe_fallback:
            debug(f"Safe load failed, retry without weights_only: {path}", dbg)
            ckpt = torch.load(path, map_location=map_location, weights_only=False)
            debug(f"Loaded checkpoint (unsafe fallback): {path}", dbg)
            return ckpt
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint '{path}': {e}")

def reconcile_checkpoint(model: torch.nn.Module, ckpt_state: dict, cfg_debug: Dict[str, Any]):
    skip = bool(cfg_debug.get("skip_mismatch_layers", True))
    loaded, skipped = [], []
    model_state = model.state_dict()
    for k, v in ckpt_state.items():
        if k not in model_state:
            skipped.append(k); continue
        if model_state[k].shape != v.shape:
            if skip:
                skipped.append(k); continue
            else:
                raise RuntimeError(f"Shape mismatch for {k}: ckpt={tuple(v.shape)} model={tuple(model_state[k].shape)}")
        model_state[k].copy_(v)
        loaded.append(k)
    model.load_state_dict(model_state)
    return loaded, skipped

def print_environment(dbg: bool = True):
    if not dbg: return
    debug(f"Python: {platform.python_version()}", True)
    debug(f"Torch:  {torch.__version__}", True)
    debug(f"CUDA available: {torch.cuda.is_available()}", True)
    if torch.cuda.is_available():
        debug(f"CUDA devices: {torch.cuda.device_count()}", True)
        debug(f"Device name: {torch.cuda.get_device_name(0)}", True)
    debug(f"Working dir: {os.getcwd()}", True)


import os
import sys
import torch

def _ensure_src_on_path():
    # utils.py is inside landmarks_only_training/src
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

def load_unet_encoder_backbone_from_ckpt(ckpt_path, device="cpu"):
    """
    Load a checkpoint saved via MLflow/wrapper that contains yacs CfgNode in ckpt['cfg'] and a model saved
    from eye_internal_segmentor.*. We:
      - allowlist safe globals (yacs.config.CfgNode) to keep weights_only=True
      - if weights_only=True still fails, we make src importable and fall back to weights_only=False (trusted)
      - extract encoder+bottleneck weights and freeze the backbone
    """
    # Allowlist yacs.config.CfgNode to enable safe weights-only loading
    try:
        import yacs.config
        torch.serialization.add_safe_globals([yacs.config.CfgNode])
    except Exception:
        pass

    ckpt = None
    try:
        # Preferred: tensors-only load; avoids importing modules like eye_internal_segmentor.*
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception as e_safe:
        print(f"[WARN] weights_only=True load failed: {e_safe}")
        print("[INFO] Attempting trusted load with weights_only=False after making packages importable...")
        _ensure_src_on_path()
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Normalize to a state_dict
    model_state = None
    # Common MLflow structure: {'cfg': CfgNode, 'model': Module or dict, ...}
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            model_state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            model_state = ckpt["state_dict"]
        elif "model" in ckpt:
            model_obj = ckpt["model"]
            if hasattr(model_obj, "state_dict"):
                model_state = model_obj.state_dict()
            elif isinstance(model_obj, dict):
                model_state = model_obj
    # Fallbacks
    if model_state is None:
        if hasattr(ckpt, "state_dict"):
            model_state = ckpt.state_dict()
        elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            model_state = ckpt

    if model_state is None:
        raise RuntimeError("Could not resolve state_dict from checkpoint; please inspect checkpoint format.")

    # Extract only encoder + bottleneck keys from UnetUpSample: enc64/32/16/8 + conv
    encoder_parts = ["enc64", "enc32", "enc16", "enc8", "conv"]
    backbone_sd = {k: v for k, v in model_state.items() if any(k.startswith(part) for part in encoder_parts)}

    # Build backbone and load subset
    try:
        from model.unet_wrapper import UNetBackbone
    except Exception:
        from .unet_wrapper import UNetBackbone

    backbone = UNetBackbone()
    missing, unexpected = backbone.load_state_dict(backbone_sd, strict=False)
    if missing:
        print(f"[INFO] Missing keys in backbone load (expected for non-encoder layers): {missing}")
    if unexpected:
        print(f"[INFO] Unexpected keys ignored in backbone load: {unexpected}")

  #  backbone.freeze_backbone()
    return backbone