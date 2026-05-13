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

The Streamlit service receives the FastAPI service hostname from Render through `BACKEND_URL`. The app accepts either a full URL or a hostname and normalizes hostnames to HTTPS.

## Deploy Backend on Vercel

Vercel is suitable for the FastAPI backend. Streamlit should be deployed separately on Streamlit Community Cloud.

The backend entrypoint is:

```text
api/index.py
```

Deploy this repo on Vercel and add these environment variables:

```text
QDRANT_URL
QDRANT_API_KEY
GEMINI_API_KEY
OPENAI_API_KEY
SUPABASE_URL
SUPABASE_KEY
```

After deployment, set the Streamlit app's `BACKEND_URL` to your Vercel backend URL, for example:

```text
https://your-project.vercel.app
```

## Deploy Frontend on Streamlit Community Cloud

Deploy `app.py` from this repo on Streamlit Community Cloud and add these secrets:

```text
BACKEND_URL=https://your-project.vercel.app
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-publishable-key
```
