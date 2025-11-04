"""
Test GPU acceleration for Arabic embeddings.
"""
import torch
from src.embeddings.arabic_embeddings import ArabicEmbeddings

def test_gpu():
    """Test GPU acceleration for Arabic embeddings."""
    # Check if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Sample Arabic text
    arabic_texts = [
        "مرحبا بالعالم",
        "كيف حالك اليوم؟",
        "نظام استرجاع المعلومات باللغة العربية"
    ]
    
    # Test with GPU (if available)
    print("\nTesting with GPU (if available):")
    model = ArabicEmbeddings(use_gpu=True)
    embeddings = model.embed_documents(arabic_texts)
    print(f"Generated {len(embeddings)} embeddings with dimension {len(embeddings[0])}")
    
    # Test with CPU
    print("\nTesting with CPU:")
    model_cpu = ArabicEmbeddings(use_gpu=False)
    embeddings_cpu = model_cpu.embed_documents(arabic_texts)
    print(f"Generated {len(embeddings_cpu)} embeddings with dimension {len(embeddings_cpu[0])}")

if __name__ == "__main__":
    test_gpu()