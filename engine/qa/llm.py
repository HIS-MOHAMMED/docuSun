import os
from langchain_ollama import ChatOllama


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required but not set.")
    return value


def get_llm_model_name() -> str:
    return _require_env("DOCUSUN_LLM_MODEL")


def get_llm_base_url() -> str:
    return _require_env("DOCUSUN_OLLAMA_BASE_URL")


def get_llm():
    llm = ChatOllama(
        model=get_llm_model_name(),
        base_url=get_llm_base_url(),
        temperature=0,
    )
    return llm