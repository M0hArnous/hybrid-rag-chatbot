# Arabic RAG System

A Retrieval-Augmented Generation (RAG) system specifically designed for Arabic content. This system combines document retrieval with language generation to provide accurate answers based on your documents.

## Features

- Document processing for Arabic text (PDF, DOCX, TXT)
- Arabic-specific text embeddings using specialized models
- Vector storage and retrieval using FAISS
- Response generation using Arabic language models
- Web interface with both FastAPI and Streamlit options
- Support for RTL (Right-to-Left) text display

## Installation

1. Clone the repository:
```
git clone <repository-url>
cd Chatbot
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

### Processing Documents

To process documents and create embeddings:

```
python main.py process --input /path/to/documents --output /path/to/save/vectors
```

### Querying the System

To query the system from the command line:

```
python main.py query --query "سؤالك هنا" --vectors /path/to/vectors
```

### Running the API Server

To start the FastAPI server:

```
python main.py api
```

The API will be available at http://localhost:8000

### Running the Streamlit Interface

To start the Streamlit web interface:

```
python main.py streamlit
```

The Streamlit app will be available at http://localhost:8501

## API Endpoints

- `GET /`: Root endpoint
- `POST /upload`: Upload a document
- `POST /query`: Query the system
- `GET /status`: Get system status

## Directory Structure

```
Chatbot/
├── data/
│   ├── raw/         # Raw documents
│   ├── processed/   # Processed documents
│   └── vectors/     # Vector embeddings
├── src/
│   ├── data/        # Document processing
│   ├── embeddings/  # Embedding generation
│   ├── retrieval/   # Document retrieval
│   ├── generation/  # Response generation
│   ├── utils/       # Utility functions
│   └── web/         # Web interfaces
└── main.py          # Main script
```

## Components

1. **Document Processing**: Handles loading and preprocessing of Arabic documents
2. **Embeddings**: Generates vector embeddings for Arabic text
3. **Vector Store**: Stores and retrieves document embeddings
4. **Retrieval**: Finds relevant documents for a query
5. **Generation**: Generates responses based on retrieved documents
6. **Web Interface**: Provides user interfaces for interacting with the system

## Arabic Language Support

This system is specifically designed for Arabic content with:

- Arabic text normalization
- Support for Arabic-specific models
- RTL text display in the web interface
- Arabic-specific text processing

## License

[MIT License](LICENSE)