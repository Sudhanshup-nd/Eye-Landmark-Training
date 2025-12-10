import torch
import torch.nn as nn
from typing import Dict

from landmarks_only_training.models.backbones.unet_wrapper import UNetBackbone



class EyeLandmarkUNetModel(nn.Module):
    def __init__(self,
                 hidden_landmarks: int = 128,
                 dropout: float = 0.0,
                 num_landmarks: int = 6,
                 use_aux_head: bool = False,
                 backbone: nn.Module =  UNetBackbone()):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.use_aux_head = use_aux_head

        self.backbone = backbone if backbone is not None else UNetBackbone()
        in_features = self.backbone.out_channels

        self.landmark_head = nn.Sequential(
            nn.Linear(in_features, hidden_landmarks),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_landmarks, num_landmarks * 2)
        )

        self.aux_head = None
        if self.use_aux_head:
            self.aux_head = nn.Sequential(
                nn.Linear(in_features, hidden_landmarks // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_landmarks // 2, num_landmarks)
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.backbone(x)
        lmks = self.landmark_head(feats).view(-1, self.num_landmarks, 2)
        out = {"landmarks": lmks}
        if self.aux_head is not None:
            out["aux"] = self.aux_head(feats)
        return out