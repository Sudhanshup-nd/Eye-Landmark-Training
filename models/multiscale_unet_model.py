import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from landmarks_only_training.models.backbones.multiscale_unet_backbone import MultiScaleUNetBackbone

class MultiScaleEyeLandmarkUNetModel(nn.Module):
    def __init__(
        self,
        num_landmarks: int = 6,
        hidden_landmarks: int = 256,
        dropout: float = 0.3,
        backbone: nn.Module = None,
        use_final_conv: bool = True,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.use_final_conv = use_final_conv
        self.backbone = backbone if backbone is not None else MultiScaleUNetBackbone()

        # Freeze encoder
        if hasattr(self.backbone, "freeze_backbone"):
            self.backbone.freeze_backbone()
        else:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Static feature dims from UnetEnc channels
        base_dim = 16 + 16 + 24 + 32  # = 88 from enc64/32/16/8
        self.in_features = base_dim + (32 if self.use_final_conv else 0)  # +32 for final

        self.landmark_head = nn.Sequential(
            nn.Linear(self.in_features, hidden_landmarks),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_landmarks, hidden_landmarks),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_landmarks, num_landmarks * 2),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.backbone(x)
        pooled = [
            F.adaptive_avg_pool2d(feats["enc64"], (1, 1)).view(x.size(0), -1),
            F.adaptive_avg_pool2d(feats["enc32"], (1, 1)).view(x.size(0), -1),
            F.adaptive_avg_pool2d(feats["enc16"], (1, 1)).view(x.size(0), -1),
            F.adaptive_avg_pool2d(feats["enc8"],  (1, 1)).view(x.size(0), -1),
        ]
        if self.use_final_conv and "final" in feats:
            pooled.append(F.adaptive_avg_pool2d(feats["final"], (1, 1)).view(x.size(0), -1))
        x_cat = torch.cat(pooled, dim=1)
        lmks = self.landmark_head(x_cat).view(-1, self.num_landmarks, 2)
        return {"landmarks": lmks}