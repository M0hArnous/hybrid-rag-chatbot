"""
Check GPU availability for the Arabic RAG system.
"""
import torch

def check_gpu():
    """Check if GPU is available and print information."""
    print("PyTorch version:", torch.__version__)
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available)
    
    if cuda_available:
        # Get the current device
        current_device = torch.cuda.current_device()
        print("Current CUDA device:", current_device)
        
        # Get the name of the current device
        device_name = torch.cuda.get_device_name(current_device)
        print("Device name:", device_name)
        
        # Get the number of GPUs
        device_count = torch.cuda.device_count()
        print("Number of GPUs:", device_count)
        
        # Memory information
        print("\nMemory Information:")
        print(f"Total memory: {torch.cuda.get_device_properties(current_device).total_memory / 1e9:.2f} GB")
        print(f"Allocated memory: {torch.cuda.memory_allocated(current_device) / 1e9:.2f} GB")
        print(f"Cached memory: {torch.cuda.memory_reserved(current_device) / 1e9:.2f} GB")
    else:
        print("No GPU available. Using CPU.")

if __name__ == "__main__":
    check_gpu()