import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from engine.qa.llm import get_llm
from engine.qa.prompt import get_prompt
from engine.retrieval.context_retriever import retrieve_context
from engine.ingestion.sources import discover_files
from engine.ingestion.loaders import load_pdf
from engine.ingestion.splitter import split_documents 
from engine.embedding.encoder import Encoder
from engine.store.chroma_store import get_retriever, load_retriever

EMBEDDING_MODEL_NAME = "google/embeddinggemma-300m"
#load environment variables from .evn file 
load_dotenv()


def index_documents(
    data_path: str = "data",
    chunk_size: int = 400,
    top_k: int = 3,
    persist_directory: str = "chroma_db",
):
    paths = discover_files(data_path)
    documents = load_pdf(paths)
    chunks = list(split_documents(chunk_size, documents, EMBEDDING_MODEL_NAME))
    encoder = Encoder()
    retriever = get_retriever(
        chunks,
        encoder,
        top_k=top_k,
        persist_directory=persist_directory,
    )
    return retriever


def query_documents(
    question: str,
    top_k: int = 3,
    persist_directory: str = "chroma_db",
):
    encoder = Encoder()
    retriever = load_retriever(
        encoder,
        top_k=top_k,
        persist_directory=persist_directory,
    )
    chain = get_prompt() | get_llm() | StrOutputParser()
    context = retrieve_context(
            question, retriever=retriever,
        )
    response = chain.invoke({"context": context, "question": question})
    return response
