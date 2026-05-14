import json
import os
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI


def _clean_json_payload(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    return cleaned


def _build_prompt(query: str, docs: list[dict[str, Any]]) -> str:
    return (
        "You are an expert relevance ranker for RAG.\n"
        "Given one query and document chunks, rank chunks by relevance for answering the query.\n"
        "Return ONLY valid JSON with this schema:\n"
        '{"ranked_indices":[{"index":0,"score":0.0}]}\n'
        "- `index` must be an integer from the given chunk indices.\n"
        "- `score` must be a number from 0.0 to 100.0.\n"
        "- Include each chunk exactly once.\n"
        "- Sort output by score descending.\n\n"
        f"Query: {query}\n"
        f"Chunks: {json.dumps(docs, ensure_ascii=False)}"
    )


def rerank_with_gemini(
    query: str,
    retrieved_docs: list[Any],
    model_name: str | None = None,
    top_k: int | None = None,
) -> list[Any]:
    """
    Rerank retrieved documents using Gemini via GOOGLE_API_KEY.
    Returns original documents if reranking is disabled or fails.
    """
    if not retrieved_docs:
        return retrieved_docs

    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return retrieved_docs

    model = (model_name or os.environ.get("DOCUSUN_RERANKER_MODEL", "")).strip()
    if not model:
        model = "gemini-1.5-flash"

    docs_payload: list[dict[str, Any]] = []
    for i, doc in enumerate(retrieved_docs):
        docs_payload.append(
            {
                "index": i,
                "content": getattr(doc, "page_content", "") or "",
                "metadata": getattr(doc, "metadata", {}) or {},
            }
        )

    try:
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0,
        )
        response = llm.invoke(_build_prompt(query, docs_payload))
        parsed = json.loads(_clean_json_payload(getattr(response, "content", "")))
        ranked = parsed.get("ranked_indices", [])
        ranked_ids = [item.get("index") for item in ranked if isinstance(item, dict)]
        ranked_ids = [i for i in ranked_ids if isinstance(i, int) and 0 <= i < len(retrieved_docs)]

        # Ensure all docs are preserved, even if model output is incomplete.
        missing = [i for i in range(len(retrieved_docs)) if i not in ranked_ids]
        final_ids = ranked_ids + missing

        reranked = [retrieved_docs[i] for i in final_ids]
        if top_k is not None and top_k > 0:
            return reranked[:top_k]
        return reranked
    except Exception:
        # Fail open to avoid breaking QA when external API is unavailable.
        return retrieved_docs[:top_k] if top_k else retrieved_docs