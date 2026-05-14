import os

from engine.retrieval.multi_query import build_multi_query_retriever
from engine.retrieval.reranker import rerank_with_gemini


def _is_multi_query_enabled() -> bool:
    return os.environ.get("DOCUSUN_ENABLE_MULTI_QUERY", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _is_reranker_enabled() -> bool:
    return os.environ.get("DOCUSUN_ENABLE_RERANKER", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def retrieve_context(query, retriever):
    """
    Retrieves and reranks documents relevant to a given query.

    Parameters:
    - query: The search query as a string.
    - retriever: An instance of a Retriever class used to fetch initial documents.

    Returns:
    - A list of reranked documents deemed relevant to the query.

    """
    final_retriever = retriever
    if _is_multi_query_enabled():
        final_retriever = build_multi_query_retriever(retriever)

    retrieved_docs = final_retriever.invoke(query)
    if not _is_reranker_enabled():
        return retrieved_docs

    reranked_docs = rerank_with_gemini(query=query, retrieved_docs=retrieved_docs)
    return reranked_docs
