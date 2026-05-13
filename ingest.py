"""Ingest role-scoped knowledge base files into Qdrant Cloud."""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


COLLECTION_NAME = "finsolve_kb"
DATA_DIR = Path("data")
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 1536
GEMINI_EMBED_BATCH_SIZE = 50
GEMINI_EMBED_BATCH_SLEEP_SECONDS = 65
ALLOWED_ACCESS_LEVELS = {"finance", "hr", "marketing", "engineering", "general"}


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


def get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default

    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=require_env("QDRANT_URL"),
        api_key=require_env("QDRANT_API_KEY"),
    )


def ensure_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION_NAME):
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIMENSIONS,
            distance=Distance.COSINE,
        ),
    )


def iter_source_files(data_dir: Path = DATA_DIR) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {data_dir}")

    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".csv"}
    )


def get_access_level(source_path: Path, data_dir: Path = DATA_DIR) -> str:
    relative_path = source_path.relative_to(data_dir)
    if len(relative_path.parts) < 2:
        raise ValueError(
            f"{source_path} must be inside a role folder such as data/finance/"
        )

    access_level = relative_path.parts[0]
    if access_level not in ALLOWED_ACCESS_LEVELS:
        allowed = ", ".join(sorted(ALLOWED_ACCESS_LEVELS))
        raise ValueError(
            f"Unsupported access level '{access_level}' for {source_path}. "
            f"Expected one of: {allowed}"
        )

    return access_level


def split_markdown_file(source_path: Path, data_dir: Path = DATA_DIR) -> list[Document]:
    markdown_text = source_path.read_text(encoding="utf-8")
    access_level = get_access_level(source_path, data_dir)

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ],
        strip_headers=False,
    )
    section_documents = header_splitter.split_text(markdown_text)

    for document in section_documents:
        document.metadata.update(
            {
                "source": str(source_path),
                "access_level": access_level,
            }
        )

    chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunks = chunk_splitter.split_documents(section_documents)

    for chunk in chunks:
        chunk.metadata["access_level"] = access_level
        chunk.metadata.setdefault("source", str(source_path))

    return chunks


def split_csv_file(source_path: Path, data_dir: Path = DATA_DIR) -> list[Document]:
    access_level = get_access_level(source_path, data_dir)
    row_documents: list[Document] = []

    with source_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            return []

        for row_index, row in enumerate(reader, start=1):
            row_text = "\n".join(
                f"{field}: {value}"
                for field, value in row.items()
                if field is not None and value not in (None, "")
            )
            if not row_text.strip():
                continue

            row_documents.append(
                Document(
                    page_content=row_text,
                    metadata={
                        "source": str(source_path),
                        "source_type": "csv",
                        "row_index": row_index,
                        "access_level": access_level,
                    },
                )
            )

    chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunks = chunk_splitter.split_documents(row_documents)

    for chunk in chunks:
        chunk.metadata["access_level"] = access_level
        chunk.metadata.setdefault("source", str(source_path))

    return chunks


def split_source_file(source_path: Path, data_dir: Path = DATA_DIR) -> list[Document]:
    suffix = source_path.suffix.lower()

    if suffix == ".md":
        return split_markdown_file(source_path, data_dir)
    if suffix == ".csv":
        return split_csv_file(source_path, data_dir)

    return []


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    documents: list[Document] = []

    for source_file in iter_source_files(data_dir):
        chunks = split_source_file(source_file, data_dir)
        for chunk_index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = chunk_index
        documents.extend(chunks)

    return documents


def get_document_ids(documents: list[Document]) -> list[str]:
    ids: list[str] = []

    for document in documents:
        source = document.metadata.get("source", "")
        chunk_index = document.metadata.get("chunk_index", "")
        ids.append(str(uuid5(NAMESPACE_URL, f"{source}:{chunk_index}")))

    return ids


def iter_batches(
    documents: list[Document],
    ids: list[str],
    batch_size: int,
) -> list[tuple[list[Document], list[str]]]:
    return [
        (documents[index : index + batch_size], ids[index : index + batch_size])
        for index in range(0, len(documents), batch_size)
    ]


def add_documents_with_rate_limit(
    vector_store: QdrantVectorStore,
    documents: list[Document],
    ids: list[str],
) -> None:
    batch_size = get_int_env("GEMINI_EMBED_BATCH_SIZE", GEMINI_EMBED_BATCH_SIZE)
    sleep_seconds = get_int_env(
        "GEMINI_EMBED_BATCH_SLEEP_SECONDS",
        GEMINI_EMBED_BATCH_SLEEP_SECONDS,
    )

    if batch_size < 1:
        raise RuntimeError("GEMINI_EMBED_BATCH_SIZE must be greater than zero")

    batches = iter_batches(documents, ids, batch_size)

    for batch_number, (batch_documents, batch_ids) in enumerate(batches, start=1):
        print(f"Upserting batch {batch_number}/{len(batches)}...")
        vector_store.add_documents(
            documents=batch_documents,
            ids=batch_ids,
            batch_size=len(batch_documents),
        )

        if batch_number < len(batches) and sleep_seconds > 0:
            print(f"Waiting {sleep_seconds}s for Gemini embedding quota...")
            time.sleep(sleep_seconds)


def ingest() -> None:
    load_env_file()

    client = get_qdrant_client()
    ensure_collection(client)

    documents = load_documents()
    if not documents:
        print(f"No Markdown or CSV files found under {DATA_DIR}/")
        return

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=require_env("GEMINI_API_KEY"),
        output_dimensionality=EMBEDDING_DIMENSIONS,
        task_type="RETRIEVAL_DOCUMENT",
    )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    ids = get_document_ids(documents)
    add_documents_with_rate_limit(vector_store, documents, ids)

    print(f"Upserted {len(documents)} chunks into Qdrant collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    ingest()
