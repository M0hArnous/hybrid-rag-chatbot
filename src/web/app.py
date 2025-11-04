"""
Web interface for Arabic RAG system using FastAPI.
"""
import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from ..data.document_loader import ArabicDocumentLoader
from ..embeddings.arabic_embeddings import ArabicEmbeddings, EmbeddingProcessor
from ..retrieval.vector_store import ArabicVectorStore
from ..retrieval.retriever import ArabicRetriever
from ..generation.generator import ArabicGenerator, ArabicRAG


# Initialize FastAPI app
app = FastAPI(title="Arabic RAG System")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define data models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 4

class QueryResponse(BaseModel):
    query: str
    response: str
    documents: List[Dict[str, Any]]

# Global variables
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
VECTOR_DIR = os.path.join(DATA_DIR, "vectors")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Ensure directories exist
os.makedirs(VECTOR_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Initialize components
embedding_model = ArabicEmbeddings()
vector_store = None
retriever = None
generator = None
rag_system = None

def initialize_rag_system():
    """Initialize the RAG system if vector store exists."""
    global vector_store, retriever, generator, rag_system
    
    # Check if vector store exists
    if os.path.exists(os.path.join(VECTOR_DIR, "index.faiss")):
        # Load vector store
        vector_store = ArabicVectorStore.load(VECTOR_DIR, embedding_model=embedding_model)
        
        # Initialize retriever
        retriever = ArabicRetriever(vector_store)
        
        # Initialize generator
        generator = ArabicGenerator()
        
        # Initialize RAG system
        rag_system = ArabicRAG(retriever, generator)
        
        return True
    
    return False

# Initialize RAG system on startup
initialize_rag_system()

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Arabic RAG System API"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document to the system."""
    # Save file to raw directory
    file_path = os.path.join(RAW_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Process document
    loader = ArabicDocumentLoader()
    try:
        documents = loader.load_document(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing document: {str(e)}")
    
    # Generate embeddings
    processor = EmbeddingProcessor(embedding_model=embedding_model)
    processed_data = processor.process_documents(documents)
    
    # Initialize or load vector store
    global vector_store, retriever, generator, rag_system
    if vector_store is None:
        vector_store = ArabicVectorStore(embedding_model=embedding_model)
    
    # Add documents to vector store
    vector_store.add_documents(documents)
    
    # Save vector store
    vector_store.save(VECTOR_DIR)
    
    # Initialize RAG system if not already initialized
    if retriever is None or generator is None or rag_system is None:
        retriever = ArabicRetriever(vector_store)
        generator = ArabicGenerator()
        rag_system = ArabicRAG(retriever, generator)
    
    return {"message": f"Document {file.filename} uploaded and processed successfully"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG system."""
    global rag_system
    
    # Check if RAG system is initialized
    if rag_system is None:
        if not initialize_rag_system():
            raise HTTPException(status_code=400, detail="RAG system not initialized. Please upload documents first.")
    
    # Process query
    result = rag_system.query(request.query)
    
    # Format documents for response
    formatted_docs = []
    for doc in result["documents"]:
        formatted_doc = {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "score": doc.metadata.get("score", 0.0)
        }
        formatted_docs.append(formatted_doc)
    
    # Return response
    return {
        "query": result["query"],
        "response": result["response"],
        "documents": formatted_docs
    }

@app.get("/status")
async def status():
    """Get system status."""
    # Check if vector store exists
    vector_store_exists = os.path.exists(os.path.join(VECTOR_DIR, "index.faiss"))
    
    # Count documents in raw directory
    raw_docs_count = len([f for f in os.listdir(RAW_DIR) if os.path.isfile(os.path.join(RAW_DIR, f))])
    
    return {
        "status": "initialized" if vector_store_exists else "not_initialized",
        "documents_count": raw_docs_count,
        "vector_store_exists": vector_store_exists
    }

def start():
    """Start the FastAPI server."""
    uvicorn.run("src.web.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start()