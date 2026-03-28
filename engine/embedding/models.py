import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

def load_embedding_model(
    model_name,
    device="cpu",
)-> HuggingFaceEmbeddings:
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
    model_name,
)-> OpenAIEmbeddings:
    """
    Loads an embedding model from the GitHub Models API using the OpenAI-compatible endpoint.

    Parameters:
    - model_name: The name of the embedding model to load.

    Returns:
    - An instance of OpenAIEmbeddings configured for GitHub Models.

    Raises:
    - ValueError: If GITHUB_TOKEN is not configured.
    """
    try:
        github_token = os.environ.get("DocuSun_GITHUB_TOKEN")
        if not github_token:
            raise ValueError("GITHUB_TOKEN is not set. Please configure it in your environment.")

        embedding_model = OpenAIEmbeddings(
            model=model_name,
            api_key=github_token,
            base_url="https://models.inference.ai.azure.com",
        )
        return embedding_model
    except Exception as e:
        print(f"An error accurred while loading the API model: {e}")
        raise