import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required but not set.")
    return value


def _get_provider(provider: str | None = None) -> str:
    value = provider if provider is not None else os.environ.get("DOCUSUN_PROVIDER", "local")
    value = value.strip().lower()
    if value not in {"local", "api", "nvidia"}:
        raise ValueError("DOCUSUN_PROVIDER must be 'local', 'api', or 'nvidia'.")
    return value


def get_llm_model_name(model_name: str | None = None) -> str:
    if model_name is not None and model_name.strip():
        return model_name.strip()
    return _require_env("DOCUSUN_LLM_MODEL")


def get_llm_base_url() -> str:
    return _require_env("DOCUSUN_OLLAMA_BASE_URL")


def get_llm_api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if key:
        return key
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    raise ValueError("GOOGLE_API_KEY (or GEMINI_API_KEY) is required but not set.")


def get_nvidia_api_key() -> str:
    return _require_env("NVIDIA_API_KEY")


def get_llm(
    provider: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
):
    resolved_provider = _get_provider(provider)
    resolved_model_name = get_llm_model_name(model_name)

    if resolved_provider == "local":
        llm = ChatOllama(
            model=resolved_model_name,
            base_url=get_llm_base_url(),
            temperature=0,
        )
        return llm

    if resolved_provider == "api":
        llm = ChatGoogleGenerativeAI(
            model=resolved_model_name,
            google_api_key=api_key or get_llm_api_key(),
            temperature=0,
        )
        return llm

    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    llm = ChatNVIDIA(
        model=resolved_model_name,
        nvidia_api_key=api_key or get_nvidia_api_key(),
        temperature=0,
    )
    return llm
