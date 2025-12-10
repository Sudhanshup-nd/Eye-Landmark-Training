import torch
import torch.nn as nn
import torchvision.models as tvm

class EyeLandmarkModel(nn.Module):
    """
    Landmark-only model:
    Backbone (e.g., ResNet18, but with in_channels adjusted) and an MLP head.
    Expects input cropped eye images (1 channel, e.g. grayscale), already resized to image_size.
    """
    def __init__(self,
                 backbone_name: str = "resnet18",
                 pretrained: bool = True,
                 hidden_landmarks: int = 128,
                 dropout: float = 0.3,
                 num_landmarks: int = 6):
        super().__init__()
        self.num_landmarks = num_landmarks

        backbone_fn = getattr(tvm, backbone_name)
        weights = None
        if pretrained:
            weights_obj = tvm.get_model_weights(backbone_name).DEFAULT
            weights = weights_obj
        self.backbone = backbone_fn(weights=weights)

        # Modify first conv layer to take 1 channel
        # Save the old weights if available (to average them for grayscale)
        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        # If pretrained, copy and average weights across input channels
        if pretrained:
            # Old shape: [out_channels, 3, kH, kW]
            # New shape: [out_channels, 1, kH, kW]
            with torch.no_grad():
                self.backbone.conv1.weight[:, 0] = old_conv.weight.mean(dim=1)
                if old_conv.bias is not None:
                    self.backbone.conv1.bias.copy_(old_conv.bias)

        # Remove final FC
        if hasattr(self.backbone, 'fc'):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError("Unexpected backbone architecture without .fc")

        self.landmark_head = nn.Sequential(
            nn.Linear(in_features, hidden_landmarks),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_landmarks, num_landmarks * 2)
        )

    def forward(self, x):
        feats = self.backbone(x)                         # [B, in_features]
        lmks = self.landmark_head(feats)                 # [B, L*2]
        lmks = lmks.view(-1, self.num_landmarks, 2)      # [B, L, 2]
        return {"landmarks": lmks}