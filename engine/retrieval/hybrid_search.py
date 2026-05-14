import os
from typing import Any
from importlib import import_module

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma

def _get_ensemble_retriever_cls():
    for module_name in ("langchain.retrievers", "langchain_classic.retrievers"):
        try:
            module = import_module(module_name)
            return getattr(module, "EnsembleRetriever")
        except (ImportError, AttributeError):
            continue
    raise ImportError(
        "EnsembleRetriever not found. Install a compatible langchain/langchain-classic version."
    )


def _parse_bool_env(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def is_hybrid_search_enabled() -> bool:
    return _parse_bool_env("DOCUSUN_ENABLE_HYBRID_SEARCH", "false")


def _get_hybrid_weight(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError:
        return default
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


def _build_keyword_retriever(docs: list[Document], keyword_k: int) -> BM25Retriever:
    keyword_retriever = BM25Retriever.from_documents(docs)
    keyword_retriever.k = keyword_k
    return keyword_retriever


def _load_docs_from_chroma(vector_store: Chroma) -> list[Document]:
    stored = vector_store.get(include=["documents", "metadatas"])
    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []

    docs: list[Document] = []
    for i, page_content in enumerate(documents):
        metadata = metadatas[i] if i < len(metadatas) and metadatas[i] is not None else {}
        docs.append(Document(page_content=page_content, metadata=metadata))
    return docs


def get_hybrid_retriever(
    docs,
    embedding_model,
    top_k: int = 3,
    persist_directory: str = "chroma_db",
) -> Any:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=persist_directory,
    )
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

    keyword_k = max(top_k, int(os.environ.get("DOCUSUN_HYBRID_BM25_K", top_k)))
    keyword_retriever = _build_keyword_retriever(docs, keyword_k=keyword_k)

    vector_weight = _get_hybrid_weight("DOCUSUN_HYBRID_VECTOR_WEIGHT", 0.5)
    keyword_weight = _get_hybrid_weight("DOCUSUN_HYBRID_KEYWORD_WEIGHT", 0.5)
    if vector_weight == 0 and keyword_weight == 0:
        vector_weight, keyword_weight = 0.5, 0.5

    ensemble_cls = _get_ensemble_retriever_cls()
    return ensemble_cls(
        retrievers=[vector_retriever, keyword_retriever],
        weights=[vector_weight, keyword_weight],
    )


def load_hybrid_retriever(
    embedding_model,
    top_k: int = 3,
    persist_directory: str = "chroma_db",
) -> Any:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
    )
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

    docs = _load_docs_from_chroma(vector_store)
    if not docs:
        return vector_retriever

    keyword_k = max(top_k, int(os.environ.get("DOCUSUN_HYBRID_BM25_K", top_k)))
    keyword_retriever = _build_keyword_retriever(docs, keyword_k=keyword_k)

    vector_weight = _get_hybrid_weight("DOCUSUN_HYBRID_VECTOR_WEIGHT", 0.5)
    keyword_weight = _get_hybrid_weight("DOCUSUN_HYBRID_KEYWORD_WEIGHT", 0.5)
    if vector_weight == 0 and keyword_weight == 0:
        vector_weight, keyword_weight = 0.5, 0.5

    ensemble_cls = _get_ensemble_retriever_cls()
    return ensemble_cls(
        retrievers=[vector_retriever, keyword_retriever],
        weights=[vector_weight, keyword_weight],
    )
