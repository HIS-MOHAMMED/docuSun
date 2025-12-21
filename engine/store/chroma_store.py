from langchain_community.vectorstores import Chroma
from typing import Any

def get_retriever(docs, embedding_model, top_k = 3) -> Any:
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
        vector_store = Chroma.from_documents(docs,embedding_model,)
        retriever = vector_store.as_retriever(k=top_k)
        
        return retriever
    except Exception as ex:
        print(f"An error occurred while initializing the retriever:{ex}")
        raise