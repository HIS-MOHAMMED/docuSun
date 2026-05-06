from langchain_huggingface import HuggingFaceEmbeddings

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