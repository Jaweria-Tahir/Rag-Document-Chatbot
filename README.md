# Rag-Document-Chatbot
A RAG-powered Document Q&amp;A web application that lets users upload PDFs or Markdown files and ask questions with source-based answers and conversational memory.

## Features

- Upload multiple PDF/Markdown documents
- Ask questions in a chat interface
- See exactly which document/chunk each answer came from
- Follow-up questions work (conversational memory)
- Upload more documents mid-session — everything stays searchable together

## Tech Stack

- **Streamlit** — web UI
- **LangChain** — RAG pipeline orchestration
- **Ollama** — local embedding model (`nomic-embed-text`) + local LLM (`qwen2.5:1.5b`)
- **PostgreSQL + pgvector** — vector database for storing embeddings

## Setup

### 1. Install and run Ollama

Download from [ollama.com](https://ollama.com), then pull the required models:

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:1.5b
```

Make sure Ollama is running (`ollama serve` if it's not already running in the background).

### 2. Set up PostgreSQL with pgvector

- Install PostgreSQL and create a database (e.g. `ragdb`)
- Enable the pgvector extension:
  ```sql
  CREATE EXTENSION IF NOT EXISTS vector;
  ```
- If `pgvector` isn't available on your system, see the [official pgvector installation guide](https://github.com/pgvector/pgvector) for your OS.

### 3. Create your own `.env` file

In the project root, create a file named `.env`:

```
DB_PASSWORD=your_postgres_password_here
```

This file is git-ignored — you must create your own; it is not included in this repo.

### 4. Run the app

```bash
streamlit run app.py
```

Opens a browser tab at `http://localhost:8501`.

## How to Test

1. Upload a document (PDF or Markdown)
2. Wait for the "Vectorstore updated!" confirmation
3. Ask a question in the chat box that you know is covered in the document
4. Check the answer and expand "Source chunks used" to see where it came from
5. Ask a follow-up question (e.g. "what about X?") to test conversational memory
6. Upload a second, different document without restarting — confirm both documents are now searchable together

## Known Limitations

- Uses a small local LLM (`qwen2.5:1.5b`) for speed on limited hardware — this occasionally misreads dense tabular data or over-paraphrases instead of quoting facts precisely. A larger model (e.g. `phi3`, or a cloud API) improves accuracy at the cost of speed/resources.
- Chunk size (`1000` characters, `200` overlap) is tuned for general prose; dense tables may need a smaller chunk size for best retrieval accuracy.
