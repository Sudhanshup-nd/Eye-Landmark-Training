import torch
import torch.nn as nn
import math

class BaseLandmarkLoss(nn.Module):
    """
    Parent class just for interface consistency.
    All losses assume:
      pred: [B,L,2] normalized local coords
      gt:   [B,L,2]
      mask: [B,L] (1 if landmark valid)
      visibility: [B,1] (0/1)
    """
    def __init__(self, visibility_gating: bool = True, eps: float = 1e-6):
        super().__init__()
        self.visibility_gating = visibility_gating
        self.eps = eps

    def forward(self, pred, gt, mask, visibility):
        raise NotImplementedError

    def _apply_mask(self, pred, gt, mask, visibility):
        visibility = visibility.view(-1)  # [B]
        eff_mask = mask * visibility.unsqueeze(1) if self.visibility_gating else mask
        eff_mask_exp = eff_mask.unsqueeze(-1)  # [B,L,1]
        return eff_mask, eff_mask_exp

class L1LandmarkLoss(BaseLandmarkLoss):
    def forward(self, pred, gt, mask, visibility):
        eff_mask, eff_mask_exp = self._apply_mask(pred, gt, mask, visibility)
        diff = torch.abs(pred - gt) * eff_mask_exp
        valid_coords = eff_mask.sum() * 2.0
        if valid_coords.item() == 0:
            return pred.sum()*0.0, {"lmk_l1":0.0,"valid_coords":0.0}
        loss = diff.sum() / (valid_coords + self.eps)
        return loss, {"lmk_l1":loss.item(),"valid_coords":float(valid_coords.item())}

class MSELandmarkLoss(BaseLandmarkLoss):
    def forward(self, pred, gt, mask, visibility):
        eff_mask, eff_mask_exp = self._apply_mask(pred, gt, mask, visibility)
        diff = (pred - gt)**2 * eff_mask_exp
        valid_coords = eff_mask.sum() * 2.0
        if valid_coords.item() == 0:
            return pred.sum()*0.0, {"mse":0.0,"valid_coords":0.0}
        loss = diff.sum() / (valid_coords + self.eps)
        return loss, {"mse":loss.item(),"valid_coords":float(valid_coords.item())}

class SmoothL1LandmarkLoss(BaseLandmarkLoss):
    def __init__(self, visibility_gating=True, eps=1e-6, beta=1.0):
        super().__init__(visibility_gating, eps)
        self.beta = beta

    def forward(self, pred, gt, mask, visibility):
        eff_mask, eff_mask_exp = self._apply_mask(pred, gt, mask, visibility)
        diff = pred - gt
        abs_diff = torch.abs(diff)
        cond = abs_diff < self.beta
        smooth = torch.where(cond, 0.5 * (diff**2) / self.beta, abs_diff - 0.5*self.beta)
        smooth = smooth * eff_mask_exp
        valid_coords = eff_mask.sum()*2.0
        if valid_coords.item() == 0:
            return pred.sum()*0.0, {"smooth_l1":0.0,"valid_coords":0.0}
        loss = smooth.sum() / (valid_coords + self.eps)
        return loss, {"smooth_l1":loss.item(),"valid_coords":float(valid_coords.item())}

class HuberLandmarkLoss(BaseLandmarkLoss):
    def __init__(self, visibility_gating=True, eps=1e-6, delta=1.0):
        super().__init__(visibility_gating, eps)
        self.delta = delta

    def forward(self, pred, gt, mask, visibility):
        eff_mask, eff_mask_exp = self._apply_mask(pred, gt, mask, visibility)
        diff = pred - gt
        abs_diff = torch.abs(diff)
        quadratic = torch.minimum(abs_diff, torch.tensor(self.delta, device=abs_diff.device))
        linear = abs_diff - quadratic
        huber = 0.5 * quadratic**2 + self.delta * linear
        huber = huber * eff_mask_exp
        valid_coords = eff_mask.sum()*2.0
        if valid_coords.item()==0:
            return pred.sum()*0.0, {"huber":0.0,"valid_coords":0.0}
        loss = huber.sum()/(valid_coords + self.eps)
        return loss, {"huber":loss.item(),"valid_coords":float(valid_coords.item())}

class WingLandmarkLoss(BaseLandmarkLoss):
    """
    Wing Loss (used for facial landmark regression):
    loss(x) = w * log(1 + |x|/epsilon) if |x| < w
              |x| - C                     otherwise
    where C = w - w*log(1 + w/epsilon)
    """
    def __init__(self, visibility_gating=True, eps=1e-6, w=10.0, epsilon=2.0):
        super().__init__(visibility_gating, eps)
        self.w = w
        self.epsilon = epsilon
        self.C = w - w * math.log(1 + w/epsilon)

    def forward(self, pred, gt, mask, visibility):
        eff_mask, eff_mask_exp = self._apply_mask(pred, gt, mask, visibility)
        diff = pred - gt
        abs_diff = torch.abs(diff)
        part_small = self.w * torch.log(1 + abs_diff / self.epsilon)
        part_large = abs_diff - self.C
        wing = torch.where(abs_diff < self.w, part_small, part_large)
        wing = wing * eff_mask_exp
        valid_coords = eff_mask.sum()*2.0
        if valid_coords.item()==0:
            return pred.sum()*0.0, {"wing":0.0,"valid_coords":0.0}
        loss = wing.sum()/(valid_coords + self.eps)
        return loss, {"wing":loss.item(),"valid_coords":float(valid_coords.item())}

def build_landmark_loss(cfg):
    t = cfg['training'].get('landmark_loss_type', 'l1').lower()
    params = cfg['training'].get('landmark_loss', {})
    if t == 'l1':
        return L1LandmarkLoss(visibility_gating=True)
    if t == 'mse':
        return MSELandmarkLoss(visibility_gating=True)
    if t == 'smooth_l1':
        beta = params.get('beta', 1.0)
        return SmoothL1LandmarkLoss(visibility_gating=True, beta=beta)
    if t == 'huber':
        delta = params.get('huber_delta', 1.0)
        return HuberLandmarkLoss(visibility_gating=True, delta=delta)
    if t == 'wing':
        w = params.get('wing_w', 10.0)
        eps = params.get('wing_epsilon', 2.0)
        return WingLandmarkLoss(visibility_gating=True, w=w, epsilon=eps)
    raise ValueError(f"Unknown landmark_loss_type: {t}")