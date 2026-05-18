import math

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer


def get_chunk_overlap_bounds(chunk_size: int) -> tuple[int, int]:
    """Return the allowed overlap bounds as 10%-20% of chunk_size."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    min_overlap = max(1, math.ceil(chunk_size * 0.10))
    max_overlap = max(min_overlap, math.floor(chunk_size * 0.20))
    return min_overlap, max_overlap


def validate_chunk_overlap(chunk_size: int, chunk_overlap: int | None) -> int:
    """Clamp overlap into the valid 10%-20% window for the given chunk size."""
    min_overlap, max_overlap = get_chunk_overlap_bounds(chunk_size)
    if chunk_overlap is None:
        return min_overlap
    return min(max(chunk_overlap, min_overlap), max_overlap)


def split_documents(
        chunk_size: int,
        knowladge_base,
        tokenizer_name,
        chunk_overlap: int | None = None,
):
    """
    Splits the documents into chunks of maximum size "chunk_size" tokens, using specified tokenizer

    Parameters:
    - chunk_size: The maximum number of tokens for each chunk.
    - knowladge_base: A list of langChainDocument objects to be split.
    - tokenizer_name: The name of tokenizer to use.
    - chunk_overlap: Optional overlap value; clamped to 10%-20% of chunk_size.

    Returns:
    - A list of langChainDocument objects, each representing a  chunk. Duplicates are removed based on 'page.content'.

    Riases:
    - ImportError: If necessary modules for tokenization are not avaiable.
    """

    resolved_chunk_overlap = validate_chunk_overlap(chunk_size, chunk_overlap)

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=resolved_chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
    )

    docs_processed = (text_splitter.split_documents([doc]) for doc in knowladge_base)

    # Flatten list and remove duplicates more efficiently
    unique_texts = set()
    docs_processed_unique = []
    for doc_chunk in docs_processed:
        for doc in doc_chunk:
            if doc.page_content not in unique_texts:
                unique_texts.add(doc.page_content)
                docs_processed_unique.append(doc)

    return docs_processed_unique
