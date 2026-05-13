import os
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required but not set.")
    return value


def _get_provider() -> str:
    value = os.environ.get("DOCUSUN_PROVIDER", "local").strip().lower()
    if value not in {"local", "api"}:
        raise ValueError("DOCUSUN_PROVIDER must be 'local' or 'api'.")
    return value


def get_llm_model_name() -> str:
    return _require_env("DOCUSUN_LLM_MODEL")


def get_llm_base_url() -> str:
    return _require_env("DOCUSUN_OLLAMA_BASE_URL")


def get_llm_api_key() -> str:
    return _require_env("GOOGLE_API_KEY")


def get_llm():
    provider = _get_provider()
    if provider == "local":
        llm = ChatOllama(
            model=get_llm_model_name(),
            base_url=get_llm_base_url(),
            temperature=0,
        )
        return llm
    llm = ChatGoogleGenerativeAI(
        model=get_llm_model_name(),
        google_api_key=get_llm_api_key(),
        temperature=0,
    )
    return llm