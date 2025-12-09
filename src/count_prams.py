from .new_unet_model import EyeLandmarkUNetModel

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total:,}')
    print(f'Trainable parameters: {trainable:,}')

# Example of usage
model =  EyeLandmarkUNetModel(
    hidden_landmarks=128,
    dropout=0.3,
    num_landmarks=6,
    use_aux_head=False  # or False, as needed
)
count_parameters(model)