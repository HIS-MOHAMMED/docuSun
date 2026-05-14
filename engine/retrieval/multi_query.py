import os
from importlib import import_module
from typing import Any

from langchain_cohere import ChatCohere
from pydantic import SecretStr


def _get_cohere_model_name() -> str:
    return os.environ.get("DOCUSUN_MULTI_QUERY_MODEL", "command-r").strip()


def build_multi_query_retriever(base_retriever: Any) -> Any:
    """
    Build a MultiQueryRetriever powered by Cohere.
    Falls back to the provided retriever when setup fails.
    """
    api_key = os.environ.get("COHERE_API_KEY", "").strip()
    if not api_key:
        return base_retriever

    try:
        # Keep import lazy to avoid hard import errors when langchain is absent
        # or when package layout differs across versions.
        try:
            multi_query_mod = import_module("langchain.retrievers.multi_query")
        except Exception:
            multi_query_mod = import_module("langchain_classic.retrievers.multi_query")
        MultiQueryRetriever = getattr(multi_query_mod, "MultiQueryRetriever")

        llm = ChatCohere(
            model=_get_cohere_model_name(),
            cohere_api_key=SecretStr(api_key),
            temperature=0,
        )
        return MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm,
        )
    except Exception:
        return base_retriever
