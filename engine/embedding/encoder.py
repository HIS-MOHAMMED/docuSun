from typing import Any, List, Sequence, Optional
from engine.embedding.models import load_embedding_model, load_embedding_model_api


class Encoder:
    """
    A simple wrapper around embedding models that provides
    a clean interface for embedding texts and queries.
    """

    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        model_name: str = "text-embedding-3-small",
        provider: str = "api",
        device: str = "cpu",
    ):
        """
        Initialize the encoder. If an embedding model is not provided,
        load one using the selected provider.

        Parameters:
        - provider: "api" for GitHub Models endpoint, "local" for HuggingFace local model.
        """
        provider = (provider or "api").strip().lower()
        if provider not in {"api", "local"}:
            raise ValueError("Invalid embedding provider. Use 'api' or 'local'.")

        if embedding_model:
            self.embedding_model = embedding_model
        elif provider == "local":
            self.embedding_model = load_embedding_model(
                model_name=model_name,
                device=device,
            )
        else:
            self.embedding_model = load_embedding_model_api(
                model_name=model_name,
            )

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Embed a list of texts (chunks).

        Parameters
        - texts : Sequence[str]

        Returns
        - List[List[float]] : A list of embedding vectors
        """
        return self.embedding_model.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string.

        Parameters:
        - query : str

        Returns:
        - List[float] : The embedding vector for the query
        """
        query = (query or "").strip()
        if not query:
            raise ValueError("Query is empty and cannot be embedded.")

        return self.embedding_model.embed_query(query)