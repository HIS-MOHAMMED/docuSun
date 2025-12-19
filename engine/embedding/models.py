from langchain_huggingface import HuggingFaceEmbeddings

def load_embedding_model(
        model_name,
        device
)-> HuggingFaceEmbeddings:
    """
    Loads an embedding model from the Hugging Face repository with specified configurations.

    Parameters:
    - model_name: The name of the model to load. Defaults to "BAAI/bge-large-en-v1.5"
    - device: The device to run the model on (e.g., 'cpu', 'cuda', 'mps').Defaults to 'cpu'

    Returns:
    - An instance of HuggingFaceBgeEmbeddings configured with the specified model and device.

    Raises:
    - ValueError: If an unsupported device is specified.
    - OSError: If the model cannot be loaded from the Hugging Face repository.
    """

    model_kwargs = {"device":device}
    encode_kwargs = {"normalize_embeddings": True}  # For cosine similarity computation


    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs = encode_kwargs
        )
        return embedding_model
    except Exception as e:
        print(f"An error accurred while loading the model: {e}")
        raise