"""
Vector store for Arabic RAG system.
Uses FAISS for efficient similarity search and retrieval.
"""
import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from src.embeddings.arabic_embeddings import ArabicEmbeddings


class ArabicVectorStore:
    """Vector store for Arabic documents using FAISS."""

    def __init__(self, embedding_model: Optional[ArabicEmbeddings] = None):
        """
        Initialize the vector store.
        
        Args:
            embedding_model: Arabic embedding model to use
        """
        self.embedding_model = embedding_model or ArabicEmbeddings()
        self.index = None
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []

    # ---------------------------------------------------------
    # 🧠 ADD DOCUMENTS
    # ---------------------------------------------------------
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the FAISS vector store."""
        if not documents:
            print("⚠️ No documents provided to add.")
            return

        texts = [doc.page_content.strip() for doc in documents if doc.page_content.strip()]
        metadatas = [doc.metadata for doc in documents]

        if not texts:
            print("⚠️ No valid text found in provided documents.")
            return

        print(f"[INFO] Generating embeddings for {len(texts)} Arabic documents...")
        embeddings = self.embedding_model.embed_documents(texts)

        if not embeddings or len(embeddings) == 0:
            raise ValueError("❌ No embeddings were generated — check your embedding model or input texts.")

        embedding_dim = len(embeddings[0])
        if self.index is None:
            print(f"[INFO] Creating new FAISS index with dimension {embedding_dim}")
            self.index = faiss.IndexFlatL2(embedding_dim)

        self.index.add(np.array(embeddings, dtype=np.float32))
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        print(f"[INFO] Added {len(texts)} documents to the FAISS index.")

    # ---------------------------------------------------------
    # 🔍 SIMILARITY SEARCH
    # ---------------------------------------------------------
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search for the given query.
        Returns top-k matching Arabic documents.
        """
        if self.index is None or not self.texts:
            print("⚠️ Vector store is empty. Add documents first.")
            return []

        print(f"[INFO] Generating embedding for query: {query}")
        query_embedding = self.embedding_model.embed_query(query)
        query_vector = np.array([query_embedding], dtype=np.float32)

        distances, indices = self.index.search(query_vector, k=min(k, len(self.texts)))

        results: List[Document] = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts):
                doc = Document(
                    page_content=self.texts[idx],
                    metadata={**self.metadatas[idx], "score": float(distances[0][i])}
                )
                results.append(doc)

        print(f"[INFO] Retrieved {len(results)} relevant documents for query.")
        return results

    # ---------------------------------------------------------
    # 💾 SAVE / LOAD
    # ---------------------------------------------------------
    def save(self, directory: str) -> None:
        """Save FAISS index, texts, and metadata to disk."""
        os.makedirs(directory, exist_ok=True)

        if self.index is not None:
            faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
            print(f"[INFO] FAISS index saved to {directory}/index.faiss")

        with open(os.path.join(directory, "texts.json"), "w", encoding="utf-8") as f:
            json.dump(self.texts, f, ensure_ascii=False, indent=2)

        with open(os.path.join(directory, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadatas, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Vector store data successfully saved to '{directory}'")

    @classmethod
    def load(cls, directory: str, embedding_model: Optional[ArabicEmbeddings] = None) -> "ArabicVectorStore":
        """Load FAISS index, texts, and metadata from disk."""
        store = cls(embedding_model=embedding_model)
        index_path = os.path.join(directory, "index.faiss")

        if os.path.exists(index_path):
            store.index = faiss.read_index(index_path)
            print(f"[INFO] Loaded FAISS index from {index_path}")
        else:
            print(f"[WARNING] No FAISS index found at {index_path}")

        texts_path = os.path.join(directory, "texts.json")
        if os.path.exists(texts_path):
            with open(texts_path, "r", encoding="utf-8") as f:
                store.texts = json.load(f)

        metadata_path = os.path.join(directory, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                store.metadatas = json.load(f)

        print(f"[INFO] Loaded {len(store.texts)} documents into memory.")
        return store
