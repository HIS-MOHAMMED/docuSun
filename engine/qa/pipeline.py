import os
import re
from typing import Any, Callable, Optional
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from engine.qa.llm import get_llm, get_llm_model_name
from engine.qa.prompt import get_prompt
from engine.qa.reporting import (
    write_header,
    write_kv,
    write_sources,
    write_documents,
    write_chunks,
    write_similar_chunks,
    write_question_answer,
)
from engine.retrieval.context_retriever import retrieve_context
from engine.ingestion.sources import discover_files
from engine.ingestion.loaders import load_pdf
from engine.ingestion.splitter import split_documents 
from engine.embedding.encoder import Encoder
from engine.store.chroma_store import get_retriever, load_retriever

#load environment variables from .evn file 
load_dotenv()

EMBEDDING_PROVIDER = os.environ.get("DOCUSUN_EMBEDDING_PROVIDER", "api").strip().lower()
if EMBEDDING_PROVIDER not in {"api", "local"}:
    raise ValueError("DOCUSUN_EMBEDDING_PROVIDER must be either 'api' or 'local'.")

EMBEDDING_MODEL_NAME = os.environ.get(
    "DOCUSUN_EMBEDDING_MODEL",
    "text-embedding-3-small" if EMBEDDING_PROVIDER == "api" else "google/embeddinggemma-300m",
).strip()
EMBEDDING_DEVICE = os.environ.get("DOCUSUN_EMBEDDING_DEVICE", "cpu").strip()
DOCUSUN_TOKENIZER_MODEL = os.environ.get("DOCUSUN_TOKENIZER_MODEL", "gpt2").strip()


def _default_persist_directory(provider: str, model_name: str) -> str:
    safe_model = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name).strip("_")
    return f"chroma_db_{provider}_{safe_model}"


def _preview_text(text: str, limit: int = 160) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _format_chunk_previews(chunks, limit: int = 10) -> list[str]:
    previews: list[str] = []
    for i, chunk in enumerate(chunks[:limit]):
        meta = getattr(chunk, "metadata", {}) or {}
        source = meta.get("source") or meta.get("file_path") or meta.get("path") or ""
        page = meta.get("page")
        prefix = f"{i}. "
        if source:
            prefix += f"{source}"
            if page is not None:
                prefix += f" (page {page})"
            prefix += ": "
        previews.append(prefix + _preview_text(getattr(chunk, "page_content", "")))
    return previews


def _emit(log_fn: Optional[Callable[..., None]], kind: str, message: str, value: Any | None = None) -> None:
    if log_fn:
        log_fn(kind, message, value)

def index_documents(
    data_path: str = "data",
    chunk_size: int = 400,
    top_k: int = 3,
    persist_directory: str | None = None,
    log_fn: Optional[Callable[..., None]] = None,
    report_path: str | None = None,
):
    if not persist_directory:
        persist_directory = _default_persist_directory(EMBEDDING_PROVIDER, EMBEDDING_MODEL_NAME)
    _emit(log_fn, "step", "Discovering files")
    paths = discover_files(data_path)
    _emit(log_fn, "kv", "Files", len(paths))
    _emit(log_fn, "list", "File list (first 10)", paths[:10])

    _emit(log_fn, "step", "Loading documents")
    documents = load_pdf(paths)
    _emit(log_fn, "kv", "Documents loaded", len(documents))

    _emit(log_fn, "step", "Splitting into chunks")
    chunks = list(split_documents(chunk_size, documents, DOCUSUN_TOKENIZER_MODEL))
    _emit(log_fn, "kv", "Chunks created", len(chunks))
    _emit(log_fn, "chunks", "Chunks (first 10)", _format_chunk_previews(chunks, limit=10))

    _emit(log_fn, "kv", "Embedding provider", EMBEDDING_PROVIDER)
    _emit(log_fn, "kv", "Embedding model", EMBEDDING_MODEL_NAME)
    _emit(log_fn, "kv", "Persist directory", persist_directory)

    if report_path:
        with open(report_path, "w", encoding="utf-8") as out:
            write_header(out, "Document source")
            write_sources(out, paths)

            write_header(out, "Document content")
            write_documents(out, documents)

            write_header(out, "Document chunks")
            write_chunks(out, chunks)

            write_header(out, "Embedding model")
            write_kv(out, "provider", EMBEDDING_PROVIDER)
            write_kv(out, "model", EMBEDDING_MODEL_NAME)
            write_kv(out, "persist_directory", persist_directory)

    encoder = Encoder(
        model_name=EMBEDDING_MODEL_NAME,
        provider=EMBEDDING_PROVIDER,
        device=EMBEDDING_DEVICE,
    )
    _emit(log_fn, "step", "Persisting embeddings")
    retriever = get_retriever(
        chunks,
        encoder,
        top_k=top_k,
        persist_directory=persist_directory,
    )
    return retriever


def query_documents(
    question: str,
    top_k: int = 3,
    persist_directory: str | None = None,
    log_fn: Optional[Callable[..., None]] = None,
    report_path: str | None = None,
):
    if not persist_directory:
        persist_directory = _default_persist_directory(EMBEDDING_PROVIDER, EMBEDDING_MODEL_NAME)
    _emit(log_fn, "kv", "Embedding provider", EMBEDDING_PROVIDER)
    _emit(log_fn, "kv", "Embedding model", EMBEDDING_MODEL_NAME)
    _emit(log_fn, "kv", "Persist directory", persist_directory)
    _emit(log_fn, "kv", "Top k", top_k)
    _emit(log_fn, "kv", "Question", question)
    _emit(log_fn, "kv", "LLM model", get_llm_model_name())

    _emit(log_fn, "step", "Loading retriever")
    encoder = Encoder(
        model_name=EMBEDDING_MODEL_NAME,
        provider=EMBEDDING_PROVIDER,
        device=EMBEDDING_DEVICE,
    )
    retriever = load_retriever(
        encoder,
        top_k=top_k,
        persist_directory=persist_directory,
    )
    _emit(log_fn, "step", "Retrieving context")
    chain = get_prompt() | get_llm() | StrOutputParser()
    context = retrieve_context(
            question, retriever=retriever,
        )
    _emit(log_fn, "kv", "Retrieved chunks", len(context))
    _emit(log_fn, "chunks", "Similar chunks (first 10)", _format_chunk_previews(context, limit=10))

    _emit(log_fn, "step", "Generating answer")
    response = chain.invoke({"context": context, "question": question})

    if report_path:
        with open(report_path, "w", encoding="utf-8") as out:
            write_header(out, "Similar documents")
            write_similar_chunks(out, context)

            write_header(out, "The question and llm answer")
            write_question_answer(out, question, response)
    return response
