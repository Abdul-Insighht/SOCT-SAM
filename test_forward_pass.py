import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lib.enhanced_model import Enhanced_RRSIS_UOT

def test_forward():
    print("Initializing Model...")
    # Initialize the model on CPU for a quick shape test
    model = Enhanced_RRSIS_UOT(
        sam3_ckpt="./pre-trained-weights/sam3.pt",
        image_size=504,
        lora_rank=4,
        lora_alpha=8.0,
        use_dynamic_lora=True,
        use_contrastive_loss=True,
        use_multiscale_ot=True,
        use_ohem_loss=True,
        use_soct=True,
        lambda_ord=5.0,
        theta_anchor=0.1,
    )
    model = model.float()
    
    B = 2
    # Mock inputs
    images = torch.randn(B, 3, 504, 504)
    # Referring expressions with directional and non-directional relations
    captions = ["the small white plane to the left of the hangar", "a large ship on water"]
    
    print("Running Forward Pass...")
    try:
        with torch.no_grad():
            outputs = model(images, captions)
            
        print("Forward pass successful!")
        print(f"Output mask shape: {outputs['pred_masks'].shape}")
        return True
    except Exception as e:
        print(f"Forward pass failed with error: {e}")
        return False

if __name__ == "__main__":
    test_forward()
