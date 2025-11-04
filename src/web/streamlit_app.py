"""
Streamlit web interface for Arabic RAG system.
"""
import os
import sys
import streamlit as st
import requests
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.document_loader import ArabicDocumentLoader
from src.embeddings.arabic_embeddings import ArabicEmbeddings, EmbeddingProcessor
from src.retrieval.vector_store import ArabicVectorStore
from src.retrieval.retriever import ArabicRetriever
from src.generation.generator import ArabicGenerator, ArabicRAG

# Set page config
st.set_page_config(
    page_title="نظام استرجاع المعلومات العربية",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
VECTOR_DIR = os.path.join(DATA_DIR, "vectors")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Ensure directories exist
os.makedirs(VECTOR_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Initialize session state
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Custom CSS for RTL support
st.markdown("""
<style>
    body {
        direction: rtl;
    }
    .rtl {
        direction: rtl;
        text-align: right;
    }
    .document-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .source {
        font-size: 0.8em;
        color: #888;
    }
    .score {
        font-size: 0.8em;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

def initialize_rag_system():
    """Initialize the RAG system if vector store exists."""
    if os.path.exists(os.path.join(VECTOR_DIR, "index.faiss")):
        # Initialize components
        embedding_model = ArabicEmbeddings()
        vector_store = ArabicVectorStore.load(VECTOR_DIR, embedding_model=embedding_model)
        retriever = ArabicRetriever(vector_store)
        generator = ArabicGenerator()
        rag_system = ArabicRAG(retriever, generator)
        
        st.session_state.rag_system = rag_system
        return True
    
    return False

def process_document(uploaded_file):
    """Process an uploaded document."""
    # Save file to raw directory
    file_path = os.path.join(RAW_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Process document
    with st.spinner("جاري معالجة المستند..."):
        loader = ArabicDocumentLoader()
        try:
            documents = loader.load_document(file_path)
        except Exception as e:
            st.error(f"خطأ في معالجة المستند: {str(e)}")
            return False
        
        # Generate embeddings
        embedding_model = ArabicEmbeddings()
        processor = EmbeddingProcessor(embedding_model=embedding_model)
        processed_data = processor.process_documents(documents)
        
        # Initialize or load vector store
        if os.path.exists(os.path.join(VECTOR_DIR, "index.faiss")):
            vector_store = ArabicVectorStore.load(VECTOR_DIR, embedding_model=embedding_model)
        else:
            vector_store = ArabicVectorStore(embedding_model=embedding_model)
        
        # Add documents to vector store
        vector_store.add_documents(documents)
        
        # Save vector store
        vector_store.save(VECTOR_DIR)
        
        # Initialize RAG system
        retriever = ArabicRetriever(vector_store)
        generator = ArabicGenerator()
        rag_system = ArabicRAG(retriever, generator)
        
        st.session_state.rag_system = rag_system
        
        return True

def display_chat():
    """Display chat interface."""
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(f"<div class='rtl'>{message['content']}</div>", unsafe_allow_html=True)
            
            # Display sources if available
            if "documents" in message:
                st.markdown("---")
                st.markdown("<div class='rtl'><strong>المصادر:</strong></div>", unsafe_allow_html=True)
                
                for i, doc in enumerate(message["documents"]):
                    with st.expander(f"مصدر {i+1}"):
                        st.markdown(f"<div class='rtl document-box'>{doc['content']}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='rtl source'>المصدر: {doc['source']}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='rtl score'>درجة التطابق: {doc['score']:.4f}</div>", unsafe_allow_html=True)
    
    # Chat input
    if query := st.chat_input("اكتب سؤالك هنا..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(f"<div class='rtl'>{query}</div>", unsafe_allow_html=True)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("جاري التفكير..."):
                if st.session_state.rag_system:
                    result = st.session_state.rag_system.query(query)
                    
                    # Format documents for display
                    formatted_docs = []
                    for doc in result["documents"]:
                        formatted_doc = {
                            "content": doc.page_content,
                            "source": doc.metadata.get("source", "غير معروف"),
                            "score": doc.metadata.get("score", 0.0)
                        }
                        formatted_docs.append(formatted_doc)
                    
                    # Display response
                    st.markdown(f"<div class='rtl'>{result['response']}</div>", unsafe_allow_html=True)
                    
                    # Display sources
                    st.markdown("---")
                    st.markdown("<div class='rtl'><strong>المصادر:</strong></div>", unsafe_allow_html=True)
                    
                    for i, doc in enumerate(formatted_docs):
                        with st.expander(f"مصدر {i+1}"):
                            st.markdown(f"<div class='rtl document-box'>{doc['content']}</div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='rtl source'>المصدر: {doc['source']}</div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='rtl score'>درجة التطابق: {doc['score']:.4f}</div>", unsafe_allow_html=True)
                    
                    # Add assistant message to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["response"],
                        "documents": formatted_docs
                    })
                else:
                    st.error("لم يتم تهيئة النظام بعد. يرجى تحميل المستندات أولاً.")

def main():
    """Main function."""
    # Sidebar
    with st.sidebar:
        st.title("نظام استرجاع المعلومات العربية")
        st.markdown("<div class='rtl'>نظام استرجاع المعلومات المعزز بالتوليد للمحتوى العربي</div>", unsafe_allow_html=True)
        
        # File uploader
        st.subheader("تحميل المستندات")
        uploaded_files = st.file_uploader(
            "قم بتحميل ملفات PDF أو DOCX أو TXT",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("معالجة المستندات"):
                for uploaded_file in uploaded_files:
                    success = process_document(uploaded_file)
                    if success:
                        st.success(f"تم معالجة {uploaded_file.name} بنجاح")
                    else:
                        st.error(f"فشل في معالجة {uploaded_file.name}")
        
        # System status
        st.subheader("حالة النظام")
        vector_store_exists = os.path.exists(os.path.join(VECTOR_DIR, "index.faiss"))
        raw_docs_count = len([f for f in os.listdir(RAW_DIR) if os.path.isfile(os.path.join(RAW_DIR, f))])
        
        if vector_store_exists:
            st.success("النظام جاهز للاستخدام")
        else:
            st.warning("النظام غير مهيأ. يرجى تحميل المستندات أولاً.")
        
        st.info(f"عدد المستندات المحملة: {raw_docs_count}")
        
        # Clear chat button
        if st.button("مسح المحادثة"):
            st.session_state.messages = []
            st.experimental_rerun()
    
    # Main content
    st.title("نظام استرجاع المعلومات العربية")
    st.markdown("<div class='rtl'>يمكنك طرح أسئلة حول المستندات المحملة وسيقوم النظام بالإجابة عليها بناءً على المعلومات المتوفرة.</div>", unsafe_allow_html=True)
    
    # Initialize RAG system if not already initialized
    if st.session_state.rag_system is None:
        initialize_rag_system()
    
    # Display chat interface
    display_chat()

if __name__ == "__main__":
    main()