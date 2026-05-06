from typing import Any, List, Sequence, Optional
from engine.embedding.models import load_embedding_model


class Encoder:
    """
    A simple wrapper around embedding models that provides
    a clean interface for embedding texts and queries.
    """

    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        model_name: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Initialize the encoder. If an embedding model is not provided,
        load one locally using the selected model.
        """
        if embedding_model:
            self.embedding_model = embedding_model
            return
        if not model_name:
            raise ValueError("model_name is required when embedding_model is not provided.")
        self.embedding_model = load_embedding_model(
            model_name=model_name,
            device=device,
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