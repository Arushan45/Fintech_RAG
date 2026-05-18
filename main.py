"""FastAPI backend for the FinSolve RBAC RAG chatbot."""

from __future__ import annotations

import os
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpArtifacts, NlpEngine
from presidio_analyzer.predefined_recognizers import (
    CreditCardRecognizer,
    PhoneRecognizer,
)
from presidio_analyzer.recognizer_registry import RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models
import spacy

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
PII_OPERATORS = {
    "PERSON": OperatorConfig("replace", {"new_value": "[PERSON_REDACTED]"}),
    "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL_REDACTED]"}),
    "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE_REDACTED]"}),
    "CREDIT_CARD": OperatorConfig("replace", {"new_value": "[CREDIT_CARD_REDACTED]"}),
}


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


class SourceChunk(BaseModel):
    content: str
    metadata: dict[str, Any]


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class BlankEnglishNlpEngine(NlpEngine):
    """Local NLP engine that avoids runtime spaCy model downloads."""

    def __init__(self) -> None:
        self.nlp: dict[str, Any] = {}

    def load(self) -> None:
        if "en" not in self.nlp:
            self.nlp["en"] = spacy.blank("en")

    def is_loaded(self) -> bool:
        return "en" in self.nlp

    def process_text(self, text: str, language: str) -> NlpArtifacts:
        self.load()
        doc = self.nlp[language](text)
        lemmas = [token.lemma_ or token.text for token in doc]
        tokens_indices = [token.idx for token in doc]
        return NlpArtifacts(
            entities=list(doc.ents),
            tokens=doc,
            tokens_indices=tokens_indices,
            lemmas=lemmas,
            nlp_engine=self,
            language=language,
        )

    def process_batch(
        self,
        texts: Iterable[str],
        language: str,
        batch_size: int = 1,
        n_process: int = 1,
        **kwargs: Any,
    ) -> Iterator[tuple[str, NlpArtifacts]]:
        for text in texts:
            yield text, self.process_text(text, language)

    def is_stopword(self, word: str, language: str) -> bool:
        self.load()
        return self.nlp[language].vocab[word].is_stop

    def is_punct(self, word: str, language: str) -> bool:
        self.load()
        return self.nlp[language].vocab[word].is_punct

    def get_supported_entities(self) -> list[str]:
        return []

    def get_supported_languages(self) -> list[str]:
        return ["en"]


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


@lru_cache(maxsize=1)
def get_pii_analyzer() -> AnalyzerEngine:
    nlp_engine = BlankEnglishNlpEngine()
    nlp_engine.load()

    registry = RecognizerRegistry(supported_languages=["en"])
    registry.add_recognizer(
        PatternRecognizer(
            supported_entity="EMAIL_ADDRESS",
            name="local_email_recognizer",
            patterns=[
                Pattern(
                    name="email_address",
                    regex=r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
                    score=0.85,
                )
            ],
        )
    )
    registry.add_recognizer(PhoneRecognizer())
    registry.add_recognizer(CreditCardRecognizer())
    registry.add_recognizer(
        PatternRecognizer(
            supported_entity="PERSON",
            name="capitalized_full_name_recognizer",
            patterns=[
                Pattern(
                    name="capitalized_full_name",
                    regex=r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b",
                    score=0.65,
                )
            ],
            context=[
                "name",
                "employee",
                "manager",
                "customer",
                "client",
                "contact",
                "called",
                "emailed",
            ],
            global_regex_flags=0,
        )
    )
    return AnalyzerEngine(
        registry=registry,
        nlp_engine=nlp_engine,
        supported_languages=["en"],
    )


@lru_cache(maxsize=1)
def get_pii_anonymizer() -> AnonymizerEngine:
    return AnonymizerEngine()


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


def anonymize_text(text: str) -> str:
    analyzer_results = get_pii_analyzer().analyze(
        text=text,
        entities=PII_ENTITIES,
        language="en",
    )
    anonymized_result = get_pii_anonymizer().anonymize(
        text=text,
        analyzer_results=analyzer_results,
        operators=PII_OPERATORS,
    )
    return anonymized_result.text


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
    return anonymize_context_chunks(retrieved_documents)


def build_chat_chain() -> Any:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
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
) -> Iterator[str]:
    chain = build_chat_chain()
    chain_input = {
        "context": format_context(documents),
        "question": question,
    }

    try:
        for token in chain.stream(chain_input):
            if token:
                yield stream_event({"type": "token", "content": token})

        sources = [
            serialize_source(document).model_dump()
            for document in documents
        ]
        yield stream_event({"type": "sources", "sources": sources})
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
        documents = get_accessible_context_documents(
            question=request.question,
            user_role=current_user.role,
            top_k=request.top_k,
        )
        background_tasks.add_task(log_query, current_user, request.question, True)
        return StreamingResponse(
            stream_answer(request.question, documents),
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
