"""
Main entry point for the Arabic RAG System.
Supports OCR extraction, document indexing, querying, API, and Streamlit UI.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root for imports
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# ✅ Imports
from src.utils.ocr_utils import process_with_ocr
from src.data.document_loader import ArabicDocumentLoader
from src.embeddings.arabic_embeddings import ArabicEmbeddings, EmbeddingProcessor
from src.retrieval.vector_store import ArabicVectorStore
from src.retrieval.retriever import ArabicRetriever
from src.generation.generator import ArabicGenerator, ArabicRAG


# ============================================================
# 📖 OCR PROCESSING
# ============================================================
def run_ocr(input_dir: str, output_dir: str = "data/processed"):
    """
    Run OCR on all PDFs/images in a folder.
    Extracts Arabic text using Mistral OCR and saves results.
    """
    print(f"\n🔍 Starting OCR extraction from: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    input_dir = Path(input_dir)
    pdf_files = list(input_dir.glob("*.pdf"))

    if not pdf_files:
        print("⚠️ No PDF files found in input directory.")
        return

    for file_path in pdf_files:
        print(f"\n📄 Processing file: {file_path.name}")
        chunks, md_path, txt_path = process_with_ocr(file_path, output_dir=output_dir)
        print(f"✅ Saved OCR results for {file_path.name}")
        print(f"   Markdown: {md_path}")
        print(f"   Text: {txt_path}")
        print(f"   Total chunks: {len(chunks)}")


# ============================================================
# 🧠 INDEXING (Embeddings + Vector Store)
# ============================================================
def build_index(input_dir: str, output_dir: str = "data/vectors"):
    """
    Build embeddings and FAISS/Chroma vector store from processed text files.
    """
    print(f"\n📂 Starting index building from: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    loader = ArabicDocumentLoader()
    documents = loader.load_directory(input_dir)
    print(f"✅ Loaded {len(documents)} document chunks")

    embedding_model = ArabicEmbeddings()
    processor = EmbeddingProcessor(embedding_model=embedding_model)

    print("🧮 Generating embeddings...")
    processed_data = processor.process_documents(documents)
    processor.save_embeddings(processed_data, output_dir)

    vector_store = ArabicVectorStore(embedding_model=embedding_model)
    vector_store.add_documents(documents)
    vector_store.save(output_dir)

    print(f"📦 Index successfully built and saved to: {output_dir}\n")


# ============================================================
# 💬 QUERY SYSTEM
# ============================================================
def query_system(query: str, vector_dir: str):
    """
    Query the Arabic RAG system with a natural language question.
    """
    print("\n🚀 Initializing Arabic RAG System...")

    embedding_model = ArabicEmbeddings()
    vector_store = ArabicVectorStore.load(vector_dir, embedding_model=embedding_model)

    retriever = ArabicRetriever(vector_store)
    generator = ArabicGenerator()
    rag = ArabicRAG(retriever, generator)

    print(f"\n💬 Query: {query}")
    result = rag.query(query)

    print("\n🧠 Response:")
    print("=" * 50)
    print(result["response"])
    print("=" * 50)

    print("\n📚 Sources:")
    for i, doc in enumerate(result["documents"], start=1):
        content_preview = doc.page_content[:200].replace("\n", " ")
        print(f"\nSource {i}:")
        print(f"  Content: {content_preview}...")
        print(f"  File: {doc.metadata.get('source', 'Unknown')}")
        print(f"  Score: {doc.metadata.get('score', 0.0):.4f}")


# ============================================================
# 🌐 RUNNERS
# ============================================================
def run_api():
    """Run FastAPI backend."""
    print("🌐 Starting FastAPI server...")
    from src.web.app import start
    start()


def run_streamlit():
    """Run Streamlit frontend."""
    import subprocess
    app_path = Path(project_root) / "src" / "web" / "streamlit_app.py"
    print(f"🖥️ Launching Streamlit app: {app_path}")
    subprocess.run(["streamlit", "run", str(app_path)], check=True)


# ============================================================
# 🧩 CLI HANDLER
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Arabic RAG System CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # OCR Command
    p_ocr = subparsers.add_parser("ocr", help="Run OCR extraction on PDFs/images")
    p_ocr.add_argument("--input", "-i", required=True, help="Input directory of raw PDFs/images")
    p_ocr.add_argument("--output", "-o", default="data/processed", help="Output directory for extracted text")

    # Index Command
    p_index = subparsers.add_parser("index", help="Build embeddings + vector store from text files")
    p_index.add_argument("--input", "-i", required=True, help="Input directory of processed text files")
    p_index.add_argument("--output", "-o", default="data/vectors", help="Output directory for vector store")

    # Process Command
    p_process = subparsers.add_parser("process", help="Process and embed documents")
    p_process.add_argument("--input", "-i", required=True, help="Input directory of documents")
    p_process.add_argument("--output", "-o", default="data/vectors", help="Output directory for vector store")

    # Query Command
    p_query = subparsers.add_parser("query", help="Ask a question to the RAG system")
    p_query.add_argument("--query", "-q", required=True, help="Arabic query text")
    p_query.add_argument("--vectors", "-v", default="data/vectors", help="Directory of saved vectors")

    # API + Streamlit
    subparsers.add_parser("api", help="Run FastAPI backend")
    subparsers.add_parser("streamlit", help="Run Streamlit dashboard")

    args = parser.parse_args()

    if args.command == "ocr":
        run_ocr(args.input, args.output)
    elif args.command == "index":
        build_index(args.input, args.output)
    elif args.command == "process":
        build_index(args.input, args.output)  # Same as index, kept for backward compatibility
    elif args.command == "query":
        query_system(args.query, args.vectors)
    elif args.command == "api":
        run_api()
    elif args.command == "streamlit":
        run_streamlit()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
