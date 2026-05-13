# FinSolve RBAC RAG Chatbot

Production-ready Python prototype for an RBAC-aware RAG chatbot using FastAPI, Streamlit, Supabase Auth, Gemini embeddings, OpenAI `gpt-4o`, and Qdrant Cloud.

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.env` from `.env.example` and fill in the required values.

## Ingest Knowledge Base

```powershell
python ingest.py
```

## Run Locally

Start the backend:

```powershell
uvicorn main:app --host 127.0.0.1 --port 8000
```

Start the Streamlit UI:

```powershell
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open:

```text
http://127.0.0.1:8501
```

## Deploy on Render

This repo includes `render.yaml` with two services:

- `finsolve-rag-api`: FastAPI backend
- `finsolve-rag-app`: Streamlit frontend

Create a Render Blueprint from this GitHub repository and provide the required secret environment variables when prompted.
