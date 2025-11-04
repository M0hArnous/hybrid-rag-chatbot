"""
Device utilities for GPU acceleration.
"""
import torch

def get_device():
    """
    Get the appropriate device (GPU or CPU) for computation.
    
    Returns:
        torch.device: The device to use for computation
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU")
    
    return device