"""
OCR utilities for Arabic RAG system.
Uses Mistral AI for OCR processing of Arabic documents.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
from dotenv import load_dotenv

from mistralai import Mistral, DocumentURLChunk
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument

from src.utils.device_utils import get_device

# Load environment variables
load_dotenv()

# Initialize Mistral client
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("❌ Missing MISTRAL_API_KEY in environment variables.")

client = Mistral(api_key=MISTRAL_API_KEY)
device = get_device()


class TextDocument(LangchainDocument):
    """Wrapper around LangChain's Document class with default metadata support."""
    def __init__(self, page_content: str, metadata: dict = None):
        super().__init__(page_content=page_content, metadata=metadata or {})


def process_with_ocr(
    file_path: str,
    output_dir: str = "data/processed",
    chunk_size: int = 1200,
    chunk_overlap: int = 150
) -> Tuple[List[LangchainDocument], str, str]:
    """
    Process a PDF or image file using Mistral OCR.
    
    Steps:
      1. Upload file to Mistral for OCR.
      2. Extract Arabic text (with Markdown support).
      3. Save both Markdown and plain text outputs.
      4. Split into chunks for embedding.
    
    Args:
        file_path: Path to PDF or image.
        output_dir: Directory for OCR output files.
        chunk_size: Maximum text length per chunk.
        chunk_overlap: Overlap between chunks.
    
    Returns:
        (chunks, markdown_path, text_path)
    """

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"📤 Uploading file to Mistral OCR: {file_path}")

    # Upload to Mistral
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    uploaded_file = client.files.upload(
        file={"file_name": file_path.name, "content": file_bytes},
        purpose="ocr"
    )

    # Get signed URL
    signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)

    print("🧠 Running OCR with Mistral (Arabic text + images)...")
    pdf_response = client.ocr.process(
        document=DocumentURLChunk(document_url=signed_url.url),
        model="mistral-ocr-latest"
    )

    # Convert response to dict
    ocr_data = json.loads(pdf_response.model_dump_json())

    if "pages" not in ocr_data:
        raise ValueError("OCR result missing 'pages' field.")

    # Combine text from pages
    combined_text = ""
    for i, page in enumerate(ocr_data["pages"], start=1):
        page_text = page.get("markdown", "").strip()
        combined_text += f"\n\n# صفحة {i}\n{page_text}"
        print(f"🖼️ Page {i}: {len(page_text)} characters extracted")

    # Save OCR results
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    base_name = f"{file_path.stem}_{timestamp}"
    md_output = os.path.join(output_dir, f"{base_name}.md")
    txt_output = os.path.join(output_dir, f"{base_name}.txt")

    print(f"💾 Saving OCR text to:\n  • {md_output}\n  • {txt_output}")
    with open(md_output, "w", encoding="utf-8") as f_md:
        f_md.write(combined_text)
    with open(txt_output, "w", encoding="utf-8") as f_txt:
        f_txt.write(combined_text)

    # Split text into chunks
    print("✂️ Splitting text into semantic chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "،", ".", "؟", "!", ";", ":"]
    )

    document = TextDocument(
        page_content=combined_text,
        metadata={
            "source": str(file_path),
            "created_at": timestamp,
            "file_type": file_path.suffix,
            "file_name": file_path.name
        }
    )

    chunks = text_splitter.split_documents([document])

    print(f"✅ OCR processing complete: {len(chunks)} chunks created")

    return chunks, md_output, txt_output
