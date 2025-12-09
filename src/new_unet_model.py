# import torch
# import torch.nn as nn
# from typing import Dict

# # Import UNet backbone from the existing module
# try:
#     from model.unet_wrapper import UNetBackbone
# except Exception:
#     # Fallback absolute import in case workspace/module pathing differs
#     from inwdata2a.sudhanshu.model.unet_wrapper import UNetBackbone  # type: ignore


# class EyeLandmarkUNetModel(nn.Module):
#     """
#     Landmark-only model using frozen UNet backbone.
#     - Backbone outputs a feature vector; heads predict landmarks.
#     - Matches the pipeline contract: forward returns {"landmarks": [B, L, 2]}.
#     - Optionally supports an auxiliary head.
#     """
#     def __init__(self,
#                  hidden_landmarks: int = 128,
#                  dropout: float = 0.3,
#                  num_landmarks: int = 6,
#                  use_aux_head: bool = False):
#         super().__init__()
#         self.num_landmarks = num_landmarks
#         self.use_aux_head = use_aux_head

#         self.backbone = UNetBackbone()
#         # Freeze backbone by default as requested
#        # self.backbone.freeze_backbone()

#         in_features = self.backbone.out_channels  # from UNetBackbone conv channels

#         self.landmark_head = nn.Sequential(
#             nn.Linear(in_features, hidden_landmarks),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_landmarks, num_landmarks * 2)
#         )

#         if self.use_aux_head:
#             self.aux_head = nn.Sequential(
#                 nn.Linear(in_features, hidden_landmarks // 2),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(hidden_landmarks // 2, num_landmarks)
#             )
#         else:
#             self.aux_head = None

#     def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#         """
#         Expects input tensor shape [B, 1, H, W] (grayscale eye crops).
#         Returns dict with keys:
#         - "landmarks": [B, L, 2]
#         - Optional "aux": [B, L] if aux head enabled
#         """
#         feats = self.backbone(x)  # [B, C]
#         lmks = self.landmark_head(feats)  # [B, L*2]
#         lmks = lmks.view(-1, self.num_landmarks, 2)

#         out = {"landmarks": lmks}
#         if self.aux_head is not None:
#             out["aux"] = self.aux_head(feats)
#         return out





import torch
import torch.nn as nn
from typing import Dict

try:
    from model.unet_wrapper import UNetBackbone
except Exception:
    from .unet_wrapper import UNetBackbone


class EyeLandmarkUNetModel(nn.Module):
    def __init__(self,
                 hidden_landmarks: int = 128,
                 dropout: float = 0.3,
                 num_landmarks: int = 6,
                 use_aux_head: bool = False,
                 backbone: nn.Module = None):
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