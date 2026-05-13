"""FastAPI backend for the FinSolve RBAC RAG chatbot."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, status
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models

from security import CurrentUser, get_current_user


COLLECTION_NAME = "finsolve_kb"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 1536
LLM_MODEL = "gpt-4o"
DEFAULT_TOP_K = 5
SYSTEM_PROMPT = (
    "You are an AI assistant for FinSolve. Only answer using the provided context. "
    "If the answer is not in the context, state that you do not have permission "
    "or the data is unavailable."
)


def load_env_file(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE pairs from .env without overriding real env vars."""

    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value

    if "Gemini_API_KEY" in os.environ and "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = os.environ["Gemini_API_KEY"]


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} must be configured")
    return value


load_env_file()
app = FastAPI(title="FinSolve RBAC RAG Chatbot")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=10)


class SourceChunk(BaseModel):
    content: str
    metadata: dict[str, Any]


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=require_env("QDRANT_URL"),
        api_key=require_env("QDRANT_API_KEY"),
    )


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=require_env("GEMINI_API_KEY"),
        output_dimensionality=EMBEDDING_DIMENSIONS,
        task_type="RETRIEVAL_QUERY",
    )


@lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    ensure_payload_indexes()
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=COLLECTION_NAME,
        embedding=get_embeddings(),
    )


@lru_cache(maxsize=1)
def ensure_payload_indexes() -> None:
    client = get_qdrant_client()
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="metadata.access_level",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )
    except Exception as exc:
        message = str(exc).lower()
        if "already exists" not in message and "already has" not in message:
            raise


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        api_key=require_env("OPENAI_API_KEY"),
    )


def build_access_filter(user_role: str) -> models.Filter | None:
    role = user_role.strip().lower()

    if role == "c-level":
        return None

    if role == "employee":
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.access_level",
                    match=models.MatchValue(value="general"),
                )
            ]
        )

    return models.Filter(
        should=[
            models.FieldCondition(
                key="metadata.access_level",
                match=models.MatchValue(value=role),
            ),
            models.FieldCondition(
                key="metadata.access_level",
                match=models.MatchValue(value="general"),
            ),
        ]
    )


def format_context(documents: list[Document]) -> str:
    if not documents:
        return "No accessible context was retrieved."

    formatted_chunks: list[str] = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        access_level = document.metadata.get("access_level", "unknown")
        formatted_chunks.append(
            f"[Chunk {index} | source: {source} | access_level: {access_level}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(formatted_chunks)


def serialize_source(document: Document) -> SourceChunk:
    return SourceChunk(
        content=document.page_content,
        metadata=dict(document.metadata),
    )


def answer_question(question: str, user_role: str, top_k: int) -> ChatResponse:
    access_filter = build_access_filter(user_role)
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": top_k,
            "filter": access_filter,
        }
    )
    retrieved_documents = retriever.invoke(question)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Context:\n{context}\n\nQuestion:\n{question}",
            ),
        ]
    )
    chain = prompt | get_llm() | StrOutputParser()
    answer = chain.invoke(
        {
            "context": format_context(retrieved_documents),
            "question": question,
        }
    )

    return ChatResponse(
        answer=answer,
        sources=[serialize_source(document) for document in retrieved_documents],
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    current_user: CurrentUser = Depends(get_current_user),
) -> ChatResponse:
    try:
        return answer_question(
            question=request.question,
            user_role=current_user.role,
            top_k=request.top_k,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Chat service failed. Check Qdrant, Gemini, OpenAI, and Supabase configuration.",
        ) from exc
