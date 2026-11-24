# DocuMind AI — Chat With Your Documents

DocuMind AI is a Retrieval-Augmented Generation (RAG) application that lets users ask questions about PDF documents. It retrieves relevant text passages using semantic similarity and sends only those passages to Gemini to generate a grounded answer.

## Features
- Upload text-based PDF documents
- Extract text page by page
- Split text into overlapping chunks
- Generate semantic embeddings with Sentence Transformers (`all-MiniLM-L6-v2`)
- Store/search vectors using FAISS
- Retrieve the most relevant chunks for each question
- Generate context-grounded answers using Gemini
- Display retrieved source passages and page numbers
- Warn when no document has been uploaded or no API key is supplied

## RAG Architecture

`PDF → Text Extraction → Chunking → Embeddings → FAISS Vector Index → Similarity Retrieval → Gemini LLM → Answer + Sources`

## Tech Stack
- Python
- Streamlit
- Sentence Transformers
- FAISS
- PyPDF
- Google Gemini API

## Run locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/documind-ai-rag.git
cd documind-ai-rag
```

### 2. Create and activate a virtual environment (Windows PowerShell)
```powershell
python -m venv venv
.env\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the app
```bash
streamlit run app.py
```

### 5. Use the app
- Enter your Gemini API key in the sidebar.
- Upload a text-based PDF.
- Wait for the vector index to be created.
- Ask questions in the chat box.
- Expand “Retrieved sources” to inspect the passages and page numbers used.

Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## Important notes
- The first run downloads the embedding model, so it may take a few minutes.
- Scanned/image-only PDFs need OCR and are not supported by this basic version.
- FAISS index and uploaded document are kept in the current Streamlit session; they are not permanently stored.
- Do not commit API keys or `.env` files to GitHub.
- The application uses retrieved document passages as context and instructs the LLM not to invent missing information. RAG can reduce hallucinations but cannot guarantee their complete elimination.

## Learning outcomes
This project demonstrates hallucinations, RAG, embeddings, semantic similarity, document processing, chunking, vector databases, retrieval quality, and connecting retrieved context to an LLM.
