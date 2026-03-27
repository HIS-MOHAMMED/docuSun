from langchain_chroma import Chroma
from typing import Any

DEFAULT_PERSIST_DIRECTORY = "chroma_db"


def get_retriever(
    docs,
    embedding_model,
    top_k: int = 3,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
) -> Any:
    """
    Initializes a retriever object to fetch the top_k most relevant documents based on cosine similarity

    Parameters:
    - docs: A list of documents to be indexed and retrieved.
    - embeddig_model: The embedding model to use for generating document vectors.
    - top_k: The number of top relevent documents to retriever.Defaults is 3.

    Returns:
    - A retriever object configured to retriever the top_k relevant documents.

    Raises:
    - ValueError: If any input paramenter is invalid.
    """
    if top_k < 1:
        raise ValueError("top_k must be at leaset 1.")
    
    try:
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory=persist_directory,
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
        
        return retriever
    except Exception as ex:
        print(f"An error occurred while initializing the retriever:{ex}")
        raise


def load_retriever(
    embedding_model,
    top_k: int = 3,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
) -> Any:
    """
    Loads a persisted Chroma vector store and returns a retriever.

    Parameters:
    - embedding_model: The embedding model used when indexing.
    - top_k: The number of top relevent documents to retriever. Defaults is 3.
    - persist_directory: Directory where Chroma database is stored.

    Returns:
    - A retriever object configured to retrieve top_k relevant documents.
    """
    if top_k < 1:
        raise ValueError("top_k must be at leaset 1.")

    try:
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model,
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
        
        return retriever
    except Exception as ex:
        print(f"An error occurred while loading the retriever:{ex}")
        raise