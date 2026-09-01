# 💰 Financial RAG Chatbot

A local, free Retrieval-Augmented Generation (RAG) chatbot that answers questions about company financial reports (10-K filings) using PDF documents, FAISS vector search, and a locally-run LLM via Ollama.

## Features

- Ask natural language questions about Apple, Microsoft, and Tesla's financial reports
- Runs 100% locally and free — no OpenAI API key required
- Uses HuggingFace embeddings (`sentence-transformers/all-MiniLM-L6-v2`) for semantic search
- Uses Ollama with `qwen2.5:3b` for answer generation
- Company-aware retrieval filtering (won't mix up Apple's numbers with Microsoft's)
- Built-in answer validation — flags when a generated answer's number can't be verified against the source

## Tech Stack

- **Python**
- **LangChain** — document loading, text splitting, retrieval orchestration
- **FAISS** — vector similarity search
- **Ollama** (`qwen2.5:3b`) — local LLM for answer generation
- **Streamlit** — chatbot UI
- **HuggingFace `sentence-transformers`** — text embeddings

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Ollama and pull the model
Download Ollama from [ollama.com](https://ollama.com/download), then:
```bash
ollama pull qwen2.5:3b
```

### 3. Add your PDFs
Place financial report PDFs (e.g. `apple.pdf`, `microsoft.pdf`, `tesla.pdf`) into the `data/` folder.

### 4. Build the FAISS index
```bash
python -m app.vector_store
```
This processes the PDFs, creates embeddings, and saves the index to `db/faiss_index/`.

### 5. Run the chatbot
```bash
streamlit run frontend/streamlit_app.py
```

## Example Questions

- "What was Apple's net income in 2022?"
- "What was Apple's total net sales in 2023?"
- "What was Microsoft's revenue in 2023?"
- "What was Tesla's net income in 2023?"

## Known Limitations

- Some summary figures that only appear in narrative/percentage-change form in a report (rather than as a clear total in a table) may not always retrieve correctly. When this happens, the app flags the answer with a warning that it could not be verified against the source documents, rather than presenting it as fully reliable.
- Performance depends on local hardware — response times may vary (a few seconds to ~1 minute) depending on RAM/CPU and whether the Ollama model is already loaded in memory.

## Project Structure

```
financial_rag_chatbot/
├── app/
│   ├── rag.py           # RAG pipeline (retrieval + generation + validation)
│   ├── loader.py         # PDF loading
│   ├── vector_store.py   # FAISS index creation/loading
│   ├── prompts.py        # Prompt templates
│   ├── llm.py             # Ollama LLM call
│   └── config.py          # Configuration
├── data/                  # Source PDF financial reports
├── db/faiss_index/        # Saved FAISS vector index
├── frontend/
│   └── streamlit_app.py   # Streamlit chatbot UI
└── requirements.txt
```