"""
Test GPU acceleration for Arabic embeddings.
"""
import time
import torch
from src.embeddings.arabic_embeddings import ArabicEmbeddings
from src.utils.device_utils import get_device

def test_gpu_embeddings():
    """Test GPU acceleration for Arabic embeddings."""
    # Get device
    device = get_device()
    
    # Sample Arabic text
    arabic_texts = [
        "مرحبا بالعالم",
        "كيف حالك اليوم؟",
        "هذا اختبار لتسريع وحدة معالجة الرسومات",
        "نظام استرجاع المعلومات باللغة العربية",
        "استخدام الذكاء الاصطناعي لمعالجة اللغة العربية"
    ]
    
    # Initialize embedding model
    print("\nInitializing embedding model...")
    embedding_model = ArabicEmbeddings()
    
    # Time the embedding process
    print("\nGenerating embeddings...")
    start_time = time.time()
    embeddings = embedding_model.embed_documents(arabic_texts)
    end_time = time.time()
    
    # Print results
    print(f"\nEmbedding generation complete:")
    print(f"- Device: {device}")
    print(f"- Time taken: {end_time - start_time:.2f} seconds")
    print(f"- Number of texts: {len(arabic_texts)}")
    print(f"- Embedding dimension: {len(embeddings[0])}")
    
    # Print sample embedding (first 5 dimensions)
    print(f"\nSample embedding (first 5 dimensions): {embeddings[0][:5]}")
    
    print("\nTest complete!")

if __name__ == "__main__":
    test_gpu_embeddings()