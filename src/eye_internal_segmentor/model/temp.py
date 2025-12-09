"""
Debug script to inspect UNet encoder/decoder layer outputs with shape (B, C, H, W).
Input: 1x1x64x64
Prints output at each encoder layer, bottleneck, and decoder layer.
"""

from unet_wrapper import Unet
import torch

def debug_unet_forward():
    """Test forward pass and print layer-wise shapes."""
    print("=" * 90)
    print("UNet Encoder/Decoder Layer Debug - Shape Format: (Batch, Channels, Height, Width)")
    print("=" * 90)
    
    # Initialize model
    model = Unet()
    model.eval()
    
    # Create dummy input: 1x1x64x64
    input_shape = (1, 1, 64, 64)
    x = torch.randn(input_shape)
    print(f"\n[INPUT]  Shape: {tuple(x.shape)} (B={input_shape[0]}, C={input_shape[1]}, H={input_shape[2]}, W={input_shape[3]})")
    print("-" * 90)
    
    # Manually run through forward pass to print each layer output
    with torch.no_grad():
        # Encoder
        x64, x_pool64 = model.enc64(x)
        print(f"[ENC64]  After conv: {tuple(x64.shape)}, After pool: {tuple(x_pool64.shape)}")
        
        x32, x_pool32 = model.enc32(x_pool64)
        print(f"[ENC32]  After conv: {tuple(x32.shape)}, After pool: {tuple(x_pool32.shape)}")
        
        x16, x_pool16 = model.enc16(x_pool32)
        print(f"[ENC16]  After conv: {tuple(x16.shape)}, After pool: {tuple(x_pool16.shape)}")
        
        x8, x_pool8 = model.enc8(x_pool16)
        print(f"[ENC8]   After conv: {tuple(x8.shape)}, After pool: {tuple(x_pool8.shape)}")
        
        # Bottleneck
        x_bn = model.conv(x_pool8)
        print(f"[BOTTLENECK] Shape: {tuple(x_bn.shape)}")
        print("-" * 90)
        
        # Decoder
        x_dec8 = model.dec8(x_bn, x8)
        print(f"[DEC8]   Output: {tuple(x_dec8.shape)}")
        
        x_dec16 = model.dec16(x_dec8, x16)
        print(f"[DEC16]  Output: {tuple(x_dec16.shape)}")
        
        x_dec32 = model.dec32(x_dec16, x32)
        print(f"[DEC32]  Output: {tuple(x_dec32.shape)}")
        
        x_dec64 = model.dec64(x_dec32, x64)
        print(f"[DEC64]  Output: {tuple(x_dec64.shape)}")
        
        # Segmentation head
        x_seg = model.seg2(model.seg1(x_dec64))
        print(f"[SEGHEAD] Output: {tuple(x_seg.shape)}")
        print("-" * 90)
        print(f"[FINAL OUTPUT] Shape: {tuple(x_seg.shape)}")
    
    print("=" * 90)

if __name__ == "__main__":
    print("\n")
    debug_unet_forward()