"""
Arabic Document Loader
======================
Handles Arabic document ingestion for RAG systems.
Supports PDF, DOCX, and TXT formats with optional OCR for Arabic text extraction.
"""

import os
from typing import List
from pathlib import Path
import pypdf
from docx import Document as DocxDocument
import pyarabic.araby as araby
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument

# OCR utility
from src.utils.ocr_utils import process_with_ocr


class ArabicDocumentLoader:
    """
    Loader for Arabic documents with text normalization and chunking.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the Arabic document loader.

        Args:
            chunk_size (int): Maximum characters per text chunk.
            chunk_overlap (int): Overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Text splitter configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
            is_separator_regex=False,
        )

    # -------------------------
    # Public Methods
    # -------------------------

    def load_document(self, file_path: str, use_ocr: bool = False) -> List[LangchainDocument]:
        """
        Load and process a document file (PDF, DOCX, TXT).

        Args:
            file_path (str): Path to the file.
            use_ocr (bool): Whether to use OCR for scanned PDFs.

        Returns:
            List[LangchainDocument]: List of text chunks with metadata.
        """
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()

        if not file_path.exists():
            raise FileNotFoundError(f"❌ File not found: {file_path}")

        # OCR mode for PDFs
        if use_ocr and file_extension == ".pdf":
            print(f"🧠 Using OCR for Arabic text extraction: {file_path}")
            chunks, _, _ = process_with_ocr(str(file_path))
            return chunks

        # Standard text extraction
        if file_extension == ".pdf":
            text = self._extract_pdf_text(str(file_path))
        elif file_extension == ".docx":
            text = self._extract_docx_text(str(file_path))
        elif file_extension == ".txt":
            text = self._extract_txt_text(str(file_path))
        else:
            raise ValueError(f"❌ Unsupported file type: {file_extension}")

        # Normalize Arabic text
        text = self._normalize_arabic(text)

        # Split text into LangChain Document chunks
        return self._split_text_into_documents(text, str(file_path))

    def load_directory(self, directory_path: str, use_ocr: bool = False) -> List[LangchainDocument]:
        """
        Load and process all supported documents in a directory.

        Args:
            directory_path (str): Path to directory.
            use_ocr (bool): Whether to enable OCR for Arabic PDFs.

        Returns:
            List[LangchainDocument]: Combined document chunks from all files.
        """
        directory_path = Path(directory_path)
        if not directory_path.exists():
            raise FileNotFoundError(f"❌ Directory not found: {directory_path}")

        documents = []
        supported_extensions = {".pdf", ".docx", ".txt"}

        print(f"📂 Loading Arabic documents from: {directory_path}")

        for file_path in directory_path.rglob("*"):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    docs = self.load_document(str(file_path), use_ocr=use_ocr)
                    documents.extend(docs)
                    print(f"✅ Loaded: {file_path.name} ({len(docs)} chunks)")
                except Exception as e:
                    print(f"⚠️ Error loading {file_path}: {e}")

        print(f"\n📚 Total chunks loaded: {len(documents)}")
        return documents

    # -------------------------
    # Private Helpers
    # -------------------------

    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract raw text from PDF."""
        text = ""
        try:
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n\n"
        except Exception as e:
            print(f"⚠️ PDF extraction failed for {file_path}: {e}")
        return text.strip()

    def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX."""
        try:
            doc = DocxDocument(file_path)
            return "\n\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            print(f"⚠️ DOCX extraction failed for {file_path}: {e}")
            return ""

    def _extract_txt_text(self, file_path: str) -> str:
        """Extract text from TXT file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"⚠️ TXT extraction failed for {file_path}: {e}")
            return ""

    def _normalize_arabic(self, text: str) -> str:
        """
        Normalize Arabic text for consistency.
        Removes diacritics and unifies letter forms.
        """
        if not text.strip():
            return text

        text = araby.strip_tashkeel(text)     # Remove tashkeel
        text = araby.strip_tatweel(text)      # Remove tatweel
        text = araby.normalize_hamza(text)    # Normalize Hamza forms

        # Normalize Lam-Alef variations manually
        lam_alef_forms = {"ﻻ": "لا", "لأ": "لأ", "لإ": "لإ", "لآ": "لآ"}
        for old, new in lam_alef_forms.items():
            text = text.replace(old, new)

        # Cleanup multiple spaces/newlines
        text = " ".join(text.split())
        return text.strip()

    def _split_text_into_documents(self, text: str, source_path: str) -> List[LangchainDocument]:
        """
        Split normalized text into LangChain Document chunks.

        Args:
            text (str): The text to split.
            source_path (str): Source file path.

        Returns:
            List[LangchainDocument]: List of chunks with metadata.
        """
        if not text:
            return []

        metadata = {"source": source_path}
        documents = self.text_splitter.create_documents([text], [metadata])
        return documents
