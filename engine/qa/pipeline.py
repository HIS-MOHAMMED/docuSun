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
from engine.ingestion.splitter import split_documents, validate_chunk_overlap
from engine.embedding.encoder import Encoder
from engine.store.chroma_store import get_retriever, load_retriever
from engine.retrieval.hybrid_search import (
    get_hybrid_retriever,
    load_hybrid_retriever,
    is_hybrid_search_enabled,
)

from engine.retrieval.parent_retrieval import(
    create_parent_retriever,
    is_parent_retrieval_enabled,
)

# load environment variables from .env file
load_dotenv()

def _get_provider(provider: str | None = None) -> str:
    value = provider if provider is not None else os.environ.get("DOCUSUN_PROVIDER", "local")
    value = value.strip().lower()
    if value not in {"local", "api", "nvidia"}:
        raise ValueError("DOCUSUN_PROVIDER must be 'local', 'api', or 'nvidia'.")
    return value


def _resolve_runtime_setting(
    override: str | None,
    env_name: str,
    default: str | None = None,
) -> str:
    if override is not None and override.strip():
        return override.strip()
    env_value = os.environ.get(env_name, "").strip()
    if env_value:
        return env_value
    if default is not None:
        return default
    raise ValueError(f"{env_name} is required but not set.")


def _resolve_embedding_device(device: str | None) -> str:
    resolved = device if device is not None else os.environ.get("DOCUSUN_EMBEDDING_DEVICE", "cpu")
    resolved = resolved.strip() if resolved else "cpu"
    return resolved or "cpu"


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
    chunk_overlap: int | None = None,
    top_k: int = 3,
    persist_directory: str | None = None,
    tokenizer_model: str | None = None,
    embedding_model: str | None = None,
    embedding_device: str | None = None,
    embedding_provider: str | None = None,
    embedding_api_key: str | None = None,
    log_fn: Optional[Callable[..., None]] = None,
    report_path: str | None = None,
):
    resolved_embedding_provider = _get_provider(embedding_provider)
    resolved_embedding_model = _resolve_runtime_setting(
        embedding_model,
        "DOCUSUN_EMBEDDING_MODEL",
    )
    resolved_embedding_device = _resolve_embedding_device(embedding_device)
    resolved_tokenizer_model = _resolve_runtime_setting(
        tokenizer_model,
        "DOCUSUN_TOKENIZER_MODEL",
    )
    resolved_chunk_overlap = validate_chunk_overlap(chunk_size, chunk_overlap)

    if not persist_directory:
        persist_directory = _default_persist_directory(
            resolved_embedding_provider,
            resolved_embedding_model,
        )
    _emit(log_fn, "step", "Discovering files")
    paths = discover_files(data_path)
    _emit(log_fn, "kv", "Files", len(paths))
    _emit(log_fn, "list", "File list (first 10)", paths[:10])

    _emit(log_fn, "step", "Loading documents")
    documents = load_pdf(paths)
    _emit(log_fn, "kv", "Documents loaded", len(documents))

    _emit(log_fn, "step", "Splitting into chunks")
    chunks = list(
        split_documents(
            chunk_size,
            documents,
            resolved_tokenizer_model,
            chunk_overlap=resolved_chunk_overlap,
        )
    )
    _emit(log_fn, "kv", "Chunks created", len(chunks))
    _emit(log_fn, "chunks", "Chunks (first 10)", _format_chunk_previews(chunks, limit=10))
    _emit(log_fn, "kv", "Tokenizer model", resolved_tokenizer_model)
    _emit(log_fn, "kv", "Chunk overlap", resolved_chunk_overlap)

    _emit(log_fn, "kv", "Embedding model", resolved_embedding_model)
    _emit(log_fn, "kv", "Embedding provider", resolved_embedding_provider)
    _emit(log_fn, "kv", "Embedding device", resolved_embedding_device)
    _emit(log_fn, "kv", "Persist directory", persist_directory)
    _emit(log_fn, "kv", "Hybrid search", is_hybrid_search_enabled())

    if report_path:
        with open(report_path, "w", encoding="utf-8") as out:
            write_header(out, "Document source")
            write_sources(out, paths)

            write_header(out, "Document content")
            write_documents(out, documents)

            write_header(out, "Document chunks")
            write_chunks(out, chunks)

            write_header(out, "Embedding model")
            write_kv(out, "provider", resolved_embedding_provider)
            write_kv(out, "model", resolved_embedding_model)
            write_kv(out, "persist_directory", persist_directory)
            write_kv(out, "tokenizer", resolved_tokenizer_model)
            write_kv(out, "chunk_overlap", resolved_chunk_overlap)

    encoder = Encoder(
        model_name=resolved_embedding_model,
        device=resolved_embedding_device,
        provider=resolved_embedding_provider,
        api_key=embedding_api_key,
    )
    _emit(log_fn, "step", "Persisting embeddings")
    if is_parent_retrieval_enabled():
        retriever = create_parent_retriever(
            documents,
            resolved_embedding_model,
            "parent_retrieval",
            3,
            persist_directory
        )
    elif is_hybrid_search_enabled():
        retriever = get_hybrid_retriever(
            chunks,
            encoder,
            top_k=top_k,
            persist_directory=persist_directory,
        )
    else:
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
    embedding_model: str | None = None,
    embedding_device: str | None = None,
    embedding_provider: str | None = None,
    embedding_api_key: str | None = None,
    llm_model: str | None = None,
    llm_provider: str | None = None,
    llm_api_key: str | None = None,
    log_fn: Optional[Callable[..., None]] = None,
    report_path: str | None = None,
):
    resolved_embedding_provider = _get_provider(embedding_provider)
    resolved_embedding_model = _resolve_runtime_setting(
        embedding_model,
        "DOCUSUN_EMBEDDING_MODEL",
    )
    resolved_embedding_device = _resolve_embedding_device(embedding_device)
    resolved_llm_provider = _get_provider(llm_provider)
    resolved_llm_model = get_llm_model_name(model_name=llm_model)

    if not persist_directory:
        persist_directory = _default_persist_directory(
            resolved_embedding_provider,
            resolved_embedding_model,
        )
    _emit(log_fn, "kv", "Embedding model", resolved_embedding_model)
    _emit(log_fn, "kv", "Embedding provider", resolved_embedding_provider)
    _emit(log_fn, "kv", "Embedding device", resolved_embedding_device)
    _emit(log_fn, "kv", "Persist directory", persist_directory)
    _emit(log_fn, "kv", "Top k", top_k)
    _emit(log_fn, "kv", "Question", question)
    _emit(log_fn, "kv", "LLM model", resolved_llm_model)
    _emit(log_fn, "kv", "LLM provider", resolved_llm_provider)
    _emit(log_fn, "kv", "Hybrid search", is_hybrid_search_enabled())

    _emit(log_fn, "step", "Loading retriever")
    encoder = Encoder(
        model_name=resolved_embedding_model,
        device=resolved_embedding_device,
        provider=resolved_embedding_provider,
        api_key=embedding_api_key,
    )
    if is_hybrid_search_enabled():
        retriever = load_hybrid_retriever(
            encoder,
            top_k=top_k,
            persist_directory=persist_directory,
        )
    else:
        retriever = load_retriever(
            encoder,
            top_k=top_k,
            persist_directory=persist_directory,
        )
    _emit(log_fn, "step", "Retrieving context")
    chain = (
        get_prompt()
        | get_llm(
            provider=resolved_llm_provider,
            model_name=resolved_llm_model,
            api_key=llm_api_key,
        )
        | StrOutputParser()
    )
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
