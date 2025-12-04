from .sources import discover_files
from .loaders import load_pdf
from .splitter import split_documents


def ingest_documents(
        chuck_size: int,
        tokenizer_name,
        path:str
):
    """
    High-level ingestion pipeline that combines the sources, loaders, 
    cleaners (used inside loaders), and splitter modules.

    Parameters:
    - path : str
        Path to a directory containing PDF files.

    Returns:
    - list of Document
        A list of chunked LangChain Document objects, ready for embedding.
    """
    
    # Discover PDF files under the provided path
    files = discover_files(path)
    
    # Load and clean raw PDF contents into LangChain Documents
    documents = load_pdf(files)

    # Split documents into smaller chunk documents
    chunks = split_documents(chuck_size, documents, tokenizer_name)
    
    return chunks
