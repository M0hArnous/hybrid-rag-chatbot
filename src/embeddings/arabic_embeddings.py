"""
Arabic text embeddings module.
Uses multilingual models for generating embeddings with strong Arabic support.
"""
from typing import List, Dict, Any, Optional
import os
import numpy as np
import torch
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer


class ArabicEmbeddings(Embeddings):
    """Arabic text embeddings using multilingual models with strong Arabic support."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base", use_gpu: bool = None):
        """
        Initialize the multilingual embeddings model with Arabic support.

        Args:
            model_name: Name of the embedding model to use.
                Options include:
                - "intfloat/multilingual-e5-base" (default, best for Arabic)
                - "UBC-NLP/ARBERT"
                - "UBC-NLP/MARBERT"
                - "aubmindlab/bert-base-arabertv02"
            use_gpu: Whether to use GPU if available (defaults to True if GPU is available)
        """
        self.model_name = model_name

        # Determine device
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()

        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            print(f"Using GPU for embeddings: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU for embeddings")

        # Load the multilingual embedding model
        self.model = SentenceTransformer(model_name, device=str(self.device))

    # -----------------------------
    # ✅ Correct Prefix Fix (E5 models)
    # -----------------------------
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents with the proper E5 prefix.
        """
        prefixed_texts = [f"passage: {text}" for text in texts]
        embeddings = self.model.encode(
            prefixed_texts,
            batch_size=16,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query with the proper E5 prefix.
        """
        query_text = f"query: {text}"
        embedding = self.model.encode(
            query_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.tolist()


class EmbeddingProcessor:
    """Processes Arabic documents and generates embeddings."""

    def __init__(self, embedding_model: Optional[ArabicEmbeddings] = None):
        """
        Initialize the embedding processor.

        Args:
            embedding_model: Arabic embedding model to use
        """
        self.embedding_model = embedding_model or ArabicEmbeddings()

    def process_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Process documents and generate embeddings.

        Args:
            documents: List of documents to process

        Returns:
            Dictionary with document texts, embeddings, and metadata
        """
        if not documents:
            print("⚠️ No documents provided for embedding.")
            return {"texts": [], "embeddings": [], "metadatas": []}

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        print(f"🔹 Generating embeddings for {len(texts)} Arabic documents...")
        embeddings = self.embedding_model.embed_documents(texts)

        return {
            "texts": texts,
            "embeddings": embeddings,
            "metadatas": metadatas
        }

    def save_embeddings(self, processed_data: Dict[str, Any], output_dir: str) -> None:
        """
        Save embeddings, texts, and metadata to disk.

        Args:
            processed_data: Processed document data
            output_dir: Directory to save embeddings
        """
        os.makedirs(output_dir, exist_ok=True)

        embeddings_path = os.path.join(output_dir, "embeddings.npy")
        texts_path = os.path.join(output_dir, "texts.json")
        metadata_path = os.path.join(output_dir, "metadata.json")

        # Save embeddings as numpy array
        embeddings = np.array(processed_data["embeddings"])
        np.save(embeddings_path, embeddings)

        # Save texts and metadata
        import json
        with open(texts_path, "w", encoding="utf-8") as f:
            json.dump(processed_data["texts"], f, ensure_ascii=False, indent=2)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(processed_data["metadatas"], f, ensure_ascii=False, indent=2)

        print(f"✅ Embeddings saved successfully to: {output_dir}")
        print(f" - {len(processed_data['texts'])} documents embedded")
        print(f" - Embeddings shape: {embeddings.shape}")


# -----------------------------
# 🧪 Example Usage (for testing)
# -----------------------------
if __name__ == "__main__":
    from langchain_core.documents import Document

    docs = [
        Document(page_content="الإجراءات الجمركية تشمل الفحص، التقييم، ودفع الرسوم الجمركية."),
        Document(page_content="تبدأ عملية التخليص الجمركي بتقديم المستندات المطلوبة."),
    ]

    processor = EmbeddingProcessor()
    data = processor.process_documents(docs)
    processor.save_embeddings(data, output_dir="embeddings_data")
