"""
Test script for OpenRoute integration with Arabic RAG system.
"""
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from src.generation.generator import ArabicGenerator

# Load environment variables
load_dotenv()

def main():
    """Test OpenRoute integration."""
    # Check if OpenRoute API key is set
    openroute_api_key = os.getenv("OPENROUTE_API_KEY")
    if not openroute_api_key or openroute_api_key == "your_openroute_api_key_here":
        print("Error: Please set your OpenRoute API key in the .env file")
        return
    
    print("Testing OpenRoute integration...")
    
    # Create sample documents
    documents = [
        Document(page_content="الذكاء الاصطناعي هو فرع من فروع علوم الحاسوب يهتم بأتمتة السلوك الذكي."),
        Document(page_content="تعلم الآلة هو مجموعة فرعية من الذكاء الاصطناعي تركز على تطوير خوارزميات تتعلم من البيانات.")
    ]
    
    # Create generator with OpenRoute
    generator = ArabicGenerator(use_openroute=True)
    
    # Test query
    query = "ما هو الذكاء الاصطناعي؟"
    
    print(f"Query: {query}")
    print("Generating response using OpenRoute API...")
    
    # Generate response
    response = generator.generate_response(query, documents)
    
    print("\nResponse:")
    print(response)
    
    print("\nOpenRoute integration test completed successfully!")

if __name__ == "__main__":
    main()