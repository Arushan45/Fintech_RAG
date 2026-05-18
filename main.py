"""FastAPI backend for the FinSolve RBAC RAG chatbot."""

from __future__ import annotations

import os
import json
import logging
import re
from uuid import UUID
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models

from security import CurrentUser, get_current_user, get_supabase_admin_client


logger = logging.getLogger(__name__)

COLLECTION_NAME = "finsolve_kb"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 1536
DEFAULT_LLM_PROVIDER = "gemini"
OPENAI_LLM_MODEL = "gpt-4o"
GEMINI_LLM_MODEL = "gemini-2.5-flash"
DEFAULT_TOP_K = 5
SYSTEM_PROMPT = (
    "You are an AI assistant for FinSolve. Only answer using the provided context. "
    "If the answer is not in the context, state that you do not have permission "
    "or the data is unavailable."
)
DEFAULT_CORS_ORIGINS = [
    "https://fintechrag-tjmusafdzpry8drgmtcfud.streamlit.app",
    "http://127.0.0.1:8501",
    "http://localhost:8501",
]
PII_ENTITIES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD"]
EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\w)"
)
CREDIT_CARD_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
PERSON_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b")


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


def get_cors_origins() -> list[str]:
    origins = os.getenv("CORS_ORIGINS", "")
    configured_origins = [
        origin.strip().rstrip("/")
        for origin in origins.split(",")
        if origin.strip()
    ]
    return configured_origins or DEFAULT_CORS_ORIGINS


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=10)
    session_id: str | None = None


class SourceChunk(BaseModel):
    content: str
    metadata: dict[str, Any]


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "FinSolve RBAC RAG Chatbot API",
        "status": "ok",
        "health": "/healthz",
        "chat": "/api/chat",
    }


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
def get_llm() -> Any:
    provider = os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower()

    if provider == "openai":
        return ChatOpenAI(
            model=os.getenv("OPENAI_LLM_MODEL", OPENAI_LLM_MODEL),
            temperature=0,
            streaming=True,
            api_key=require_env("OPENAI_API_KEY"),
        )

    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_LLM_MODEL", GEMINI_LLM_MODEL),
            temperature=0,
            streaming=True,
            google_api_key=require_env("GEMINI_API_KEY"),
        )

    raise RuntimeError("LLM_PROVIDER must be either 'gemini' or 'openai'")


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


def is_luhn_valid(value: str) -> bool:
    digits = [int(char) for char in re.sub(r"\D", "", value)]
    if len(digits) < 13 or len(digits) > 19:
        return False

    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def redact_credit_card(match: re.Match[str]) -> str:
    return "[CREDIT_CARD_REDACTED]" if is_luhn_valid(match.group(0)) else match.group(0)


def anonymize_text(text: str) -> str:
    anonymized = EMAIL_PATTERN.sub("[EMAIL_REDACTED]", text)
    anonymized = CREDIT_CARD_PATTERN.sub(redact_credit_card, anonymized)
    anonymized = PHONE_PATTERN.sub("[PHONE_REDACTED]", anonymized)
    anonymized = PERSON_PATTERN.sub("[PERSON_REDACTED]", anonymized)
    return anonymized


def anonymize_context_chunks(documents: list[Document]) -> list[Document]:
    return [
        Document(
            page_content=anonymize_text(document.page_content),
            metadata=dict(document.metadata),
        )
        for document in documents
    ]


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


def validate_uuid(value: str) -> str:
    try:
        return str(UUID(value))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="session_id must be a valid UUID",
        ) from exc


def make_session_title(prompt: str) -> str:
    title = " ".join(prompt.strip().split())
    if len(title) > 60:
        title = f"{title[:57].rstrip()}..."
    return title or "New chat"


def extract_supabase_rows(response: Any) -> list[dict[str, Any]]:
    data = getattr(response, "data", None)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []


def create_chat_session(user: CurrentUser, prompt: str) -> str:
    response = get_supabase_admin_client().table("chat_sessions").insert(
        {
            "user_id": user.id,
            "title": make_session_title(prompt),
        }
    ).execute()
    rows = extract_supabase_rows(response)
    if not rows or not isinstance(rows[0].get("id"), str):
        raise RuntimeError("Failed to create chat session")
    return rows[0]["id"]


def ensure_user_owns_session(session_id: str, user: CurrentUser) -> None:
    response = get_supabase_admin_client().table("chat_sessions").select("id").eq(
        "id", session_id
    ).eq("user_id", user.id).limit(1).execute()
    if not extract_supabase_rows(response):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found",
        )


def resolve_chat_session(request: ChatRequest, user: CurrentUser) -> str:
    if request.session_id:
        session_id = validate_uuid(request.session_id)
        ensure_user_owns_session(session_id, user)
        return session_id

    return create_chat_session(user, request.question)


def fetch_chat_history(session_id: str, user: CurrentUser) -> list[BaseMessage]:
    ensure_user_owns_session(session_id, user)
    response = get_supabase_admin_client().table("chat_messages").select(
        "role, content, created_at"
    ).eq("session_id", session_id).order("created_at", desc=True).limit(6).execute()
    rows = list(reversed(extract_supabase_rows(response)))

    messages: list[BaseMessage] = []
    for row in rows:
        role = row.get("role")
        content = row.get("content")
        if not isinstance(content, str):
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def insert_chat_messages(session_id: str, user_prompt: str, assistant_answer: str) -> None:
    if not assistant_answer.strip():
        return

    try:
        get_supabase_admin_client().table("chat_messages").insert(
            [
                {
                    "session_id": session_id,
                    "role": "user",
                    "content": user_prompt,
                },
                {
                    "session_id": session_id,
                    "role": "assistant",
                    "content": assistant_answer,
                },
            ]
        ).execute()
    except Exception:
        logger.exception("Failed to write chat messages for session_id=%s", session_id)


def get_accessible_context_documents(
    question: str,
    user_role: str,
    top_k: int,
) -> list[Document]:
    access_filter = build_access_filter(user_role)
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": top_k,
            "filter": access_filter,
        }
    )
    retrieved_documents = retriever.invoke(question)
    anonymized_documents = anonymize_context_chunks(retrieved_documents)
    logger.info(
        "Retrieved %s accessible chunks for role=%s",
        len(anonymized_documents),
        user_role,
    )
    return anonymized_documents


def build_chat_chain() -> Any:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "human",
                "Context:\n{context}\n\nQuestion:\n{question}",
            ),
        ]
    )
    return prompt | get_llm() | StrOutputParser()


def stream_event(payload: dict[str, Any]) -> str:
    return json.dumps(payload) + "\n"


def stream_answer(
    question: str,
    documents: list[Document],
    chat_history: list[BaseMessage],
    session_id: str,
    background_tasks: BackgroundTasks,
) -> Iterator[str]:
    generated_parts: list[str] = []

    if not documents:
        message = (
            "I could not find accessible source documents for this question. "
            "Please check that Qdrant has been ingested and that your user role "
            "matches the document access level."
        )
        generated_parts.append(message)
        yield stream_event({"type": "token", "content": message})
        background_tasks.add_task(
            insert_chat_messages,
            session_id,
            question,
            "".join(generated_parts),
        )
        yield stream_event(
            {"type": "sources", "session_id": session_id, "sources": []}
        )
        return

    chain = build_chat_chain()
    chain_input = {
        "chat_history": chat_history,
        "context": format_context(documents),
        "question": question,
    }

    try:
        emitted_token = False
        for token in chain.stream(chain_input):
            if token:
                emitted_token = True
                token_text = str(token)
                generated_parts.append(token_text)
                yield stream_event({"type": "token", "content": token_text})

        if not emitted_token:
            fallback_answer = chain.invoke(chain_input)
            if fallback_answer:
                generated_parts.append(str(fallback_answer))
                yield stream_event(
                    {"type": "token", "content": str(fallback_answer)}
                )
            else:
                fallback_message = "I could not generate a response. Please try again."
                generated_parts.append(fallback_message)
                yield stream_event(
                    {
                        "type": "token",
                        "content": fallback_message,
                    }
                )

        sources = [
            serialize_source(document).model_dump()
            for document in documents
        ]
        background_tasks.add_task(
            insert_chat_messages,
            session_id,
            question,
            "".join(generated_parts),
        )
        yield stream_event(
            {"type": "sources", "session_id": session_id, "sources": sources}
        )
    except Exception:
        logger.exception("Chat stream failed")
        yield stream_event(
            {
                "type": "error",
                "message": "Chat stream failed. Check LLM provider configuration.",
            }
        )


def answer_question(question: str, user_role: str, top_k: int) -> ChatResponse:
    anonymized_documents = get_accessible_context_documents(
        question=question,
        user_role=user_role,
        top_k=top_k,
    )
    answer = build_chat_chain().invoke(
        {
            "chat_history": [],
            "context": format_context(anonymized_documents),
            "question": question,
        }
    )

    return ChatResponse(
        answer=answer,
        sources=[serialize_source(document) for document in anonymized_documents],
    )


def log_query(user: CurrentUser, query_text: str, successful: bool) -> None:
    try:
        get_supabase_admin_client().table("query_logs").insert(
            {
                "user_id": user.id,
                "user_role": user.role,
                "query_text": query_text,
                "successful": successful,
            }
        ).execute()
    except Exception:
        logger.exception("Failed to write query audit log")


@app.post("/api/chat")
def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: CurrentUser = Depends(get_current_user),
) -> StreamingResponse:
    try:
        session_id = resolve_chat_session(request, current_user)
        chat_history = fetch_chat_history(session_id, current_user)
        documents = get_accessible_context_documents(
            question=request.question,
            user_role=current_user.role,
            top_k=request.top_k,
        )
        background_tasks.add_task(log_query, current_user, request.question, True)
        return StreamingResponse(
            stream_answer(
                question=request.question,
                documents=documents,
                chat_history=chat_history,
                session_id=session_id,
                background_tasks=background_tasks,
            ),
            media_type="application/x-ndjson",
            background=background_tasks,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Chat service failed for authenticated user role=%s", current_user.role)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Chat service failed. Check Qdrant, Gemini, OpenAI, and Supabase configuration.",
        ) from exc
