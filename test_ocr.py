"""
Test script for OCR functionality with Arabic PDFs.
"""
import os
import sys
from dotenv import load_dotenv
from src.data.document_loader import ArabicDocumentLoader
from src.utils.ocr_utils import process_with_ocr

# Load environment variables (for Mistral API key)
load_dotenv()

def test_ocr_processing():
    """Test OCR processing with an Arabic PDF."""
    # Check if Mistral API key is set
    if not os.getenv("MISTRAL_API_KEY"):
        print("Error: MISTRAL_API_KEY environment variable not set.")
        print("Please create a .env file with your Mistral API key.")
        sys.exit(1)
    
    # Get PDF path from command line or use default
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        print("Usage: python test_ocr.py <path_to_arabic_pdf>")
        sys.exit(1)
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    print(f"Testing OCR processing with: {pdf_path}")
    
    # Method 1: Using the OCR utility directly
    print("\n=== Method 1: Direct OCR Processing ===")
    chunks, md_path, txt_path = process_with_ocr(pdf_path)
    print(f"OCR processing complete: {len(chunks)} chunks created")
    print(f"Markdown output: {md_path}")
    print(f"Text output: {txt_path}")
    
    # Method 2: Using the document loader with OCR option
    print("\n=== Method 2: Document Loader with OCR ===")
    loader = ArabicDocumentLoader()
    ocr_chunks = loader.load_document(pdf_path, use_ocr=True)
    print(f"Document loader with OCR: {len(ocr_chunks)} chunks created")
    
    # Print sample text from first chunk
    if chunks:
        print("\n=== Sample Text from First Chunk ===")
        print(chunks[0].page_content[:500])
    
    print("\nOCR testing complete!")

if __name__ == "__main__":
    test_ocr_processing()