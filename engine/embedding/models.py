import os
from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required but not set.")
    return value

def load_embedding_model(
    model_name,
    device="cpu",
) -> HuggingFaceEmbeddings:
    """
    Loads a local embedding model from Hugging Face.

    Parameters:
    - model_name: The name of the local embedding model to load.
    - device: The device to run the model on (for example: cpu, cuda, mps).

    Returns:
    - An instance of HuggingFaceEmbeddings.
    """
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}

    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        return embedding_model
    except Exception as e:
        print(f"An error accurred while loading the local model: {e}")
        raise


def load_embedding_model_api(
    model_name: str,
    api_key: str | None = None,
) -> GoogleGenerativeAIEmbeddings:
    """
    Loads an API embedding model via Gemini.

    Parameters:
    - model_name: The Gemini embedding model name.
    - api_key: Optional API key (defaults to GOOGLE_API_KEY).

    Returns:
    - An instance of GoogleGenerativeAIEmbeddings.
    """
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    else:
        _require_env("GOOGLE_API_KEY")

    try:
        embedding_model = GoogleGenerativeAIEmbeddings(model=model_name)
        return embedding_model
    except Exception as e:
        print(f"An error accurred while loading the API model: {e}")
        raise


def load_embedding_model_nvidia(
    model_name: str,
    api_key: str | None = None,
) -> Any:
    """
    Loads an embedding model via NVIDIA Build API.

    Parameters:
    - model_name: The NVIDIA embedding model name.
    - api_key: Optional API key (defaults to NVIDIA_API_KEY).
    """
    resolved_api_key = api_key or _require_env("NVIDIA_API_KEY")

    try:
        from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

        embedding_model = NVIDIAEmbeddings(
            model=model_name,
            nvidia_api_key=resolved_api_key,
            truncate="NONE",
        )
        return embedding_model
    except Exception as e:
        print(f"An error accurred while loading the NVIDIA API model: {e}")
        raise
