
"""
Run: 
python -m landmarks_only_training.scripts.count_prams

"""

from ..models.unet_encoder_model import EyeLandmarkUNetModel
from ..models.resnet_encoder_model import EyeLandmarkModel

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total:,}')
    print(f'Trainable parameters: {trainable:,}')

# Example of usage
model1 =  EyeLandmarkUNetModel(
    hidden_landmarks=128,
    dropout=0.3,
    num_landmarks=6,
    use_aux_head=False  # or False, as needed
)

model2= EyeLandmarkModel(
    backbone_name="resnet18",
    pretrained=False,
    num_landmarks=6
)

print("EyeLandmarkUNetModel parameters:")
count_parameters(model1)
print("EyeLandmarkModel parameters:")
count_parameters(model2)    