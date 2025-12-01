from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer


def split_documents(
        chunk_size: int,
        knowladge_base,
        tokenizer_name
):
    """
    Splits the documents into chunks of maximum size "chunk_size" tokens, using specified tokenizer

    Parameters:
    - chunk_size: The maximum number of tokens for each chunk.
    - knowladge_base: A list of langChainDocument objects to be split.
    - tokenizer_name: The name of tokenizer to use.

    Returns:
    - A list of langChainDocument objects, each representing a  chunk. Duplicates are removed based on 'page.content'.

    Riases:
    - ImportError: If necessary modules for tokenization are not avaiable.
    """

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size = chunk_size,
        chunk_overlap = int(chunk_size / 10),
        add_start_index = True,
        strip_whitespace = True,    
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

