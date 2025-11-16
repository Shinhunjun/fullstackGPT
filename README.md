# FullStackGPT - LangChain Web Applications

A collection of AI-powered web applications built with LangChain and Streamlit, demonstrating various use cases of Large Language Models (LLMs) for document analysis, quiz generation, and private AI assistants.

## Features

### 1. DocumentGPT
- Upload PDF, TXT, or DOCX files
- AI-powered document Q&A using RAG (Retrieval-Augmented Generation)
- Vector embeddings with FAISS for semantic search
- Real-time streaming responses

### 2. QuizGPT
- Automatic quiz generation from uploaded documents
- Multiple choice questions based on document content
- Interactive quiz interface with scoring

### 3. PrivateGPT
- Local LLM integration for private document analysis
- No data sent to external servers
- Secure and confidential document processing

## Tech Stack

- **Framework**: Streamlit
- **LLM Integration**: LangChain, OpenAI GPT
- **Vector Database**: FAISS
- **Embeddings**: OpenAI Embeddings with caching
- **Document Processing**: UnstructuredFileLoader
- **Language**: Python 3.11

## Project Structure

```
fullstackGPT/
├── Home1.py                    # Main entry point
├── home.py                     # Homepage with demo
├── pages/
│   ├── 01_DocumentGPT.py      # Document Q&A application
│   ├── 02_QuizGPT.py          # Quiz generation application
│   └── 03_PrivateGPT.py       # Private LLM application
├── utils.py                    # Shared utility functions
├── notebook.ipynb              # Development notebook
├── requirements.txt            # Python dependencies
├── prompt.json                 # Prompt templates (JSON)
├── prompt.yaml                 # Prompt templates (YAML)
├── cache.db                    # Local cache database
└── .cache/                     # Embeddings cache directory
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Shinhunjun/fullstackGPT.git
   cd fullstackGPT
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

4. Run the application:
   ```bash
   streamlit run Home1.py
   ```

## Usage

1. **DocumentGPT**: Upload a document and ask questions about its content
2. **QuizGPT**: Upload a document to generate interactive quizzes
3. **PrivateGPT**: Use local LLM for confidential document analysis

## Key Technologies

- **LangChain**: Framework for building LLM applications
- **Streamlit**: Interactive web application framework
- **FAISS**: Facebook AI Similarity Search for vector storage
- **OpenAI**: GPT models for natural language understanding
- **RAG**: Retrieval-Augmented Generation for accurate responses

## Author

Hunjun Shin

## License

This project is for educational purposes as part of Northeastern University coursework.
