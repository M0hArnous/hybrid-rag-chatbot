"""
ArabicRetriever: Retriever for Arabic RAG systems.
Handles query normalization, similarity search, and debugging visualization.
"""

from typing import List
import pyarabic.araby as araby
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, PrivateAttr

from ..retrieval.vector_store import ArabicVectorStore


class ArabicRetriever(BaseRetriever):
    """Retriever for Arabic documents."""

    _vector_store: ArabicVectorStore = PrivateAttr()
    k: int = Field(default=4, description="Number of documents to retrieve")

    def __init__(self, vector_store: ArabicVectorStore, top_k: int = 4):
        """
        Initialize the Arabic retriever.

        Args:
            vector_store: Vector store for document retrieval
            top_k: Number of documents to retrieve
        """
        super().__init__()
        self._vector_store = vector_store
        self.k = top_k

    # -----------------------------------------------------
    # 🧹 Text Normalization
    # -----------------------------------------------------
    def _normalize_query(self, query: str) -> str:
        """
        Normalize Arabic query text to improve matching accuracy.
        Removes tashkeel, tatweel, and unifies common Arabic variants.
        """
        if not query:
            return ""

        # Remove diacritics and elongation
        query = araby.strip_tashkeel(query)
        query = araby.strip_tatweel(query)

        # Normalize Hamza variants
        query = query.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")

        # Normalize Taa Marbuta and Alef Maqsura
        query = query.replace("ة", "ه").replace("ى", "ي")

        # Remove extra whitespace
        query = " ".join(query.split())

        return query

    # -----------------------------------------------------
    # 🔍 Document Retrieval
    # -----------------------------------------------------
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve documents relevant to a given Arabic query.

        Args:
            query: User query (Arabic)

        Returns:
            List of retrieved documents
        """
        # Normalize input
        normalized_query = self._normalize_query(query)

        print("\n============================")
        print(f"🔍 Original Query: {query}")
        print(f"🧠 Normalized Query: {normalized_query}")
        print("============================")

        # Retrieve top-k documents
        documents = self._vector_store.similarity_search(normalized_query, k=self.k)

        if not documents:
            print("⚠️ No documents found for this query.")
            return []

        # Debug preview
        for i, doc in enumerate(documents):
            preview = doc.page_content[:300].replace("\n", " ")
            print(f"\n📄 Document {i + 1}:")
            print(f"→ Metadata: {doc.metadata}")
            print(f"→ Preview: {preview}...")
            print("----------------------------")

        return documents

    # -----------------------------------------------------
    # ⚡ Async version
    # -----------------------------------------------------
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of get_relevant_documents."""
        return self.get_relevant_documents(query)
