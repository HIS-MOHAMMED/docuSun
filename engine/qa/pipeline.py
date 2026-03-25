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
from engine.store.chroma_store import get_retriever

EMBEDDING_MODEL_NAME = "google/embeddinggemma-300m"
#load environment variables from .evn file 
load_dotenv()
def get_answer(context, question):
    paths = discover_files("data")
    documents = load_pdf(paths)
    chunks = list(split_documents(400, documents, EMBEDDING_MODEL_NAME))
    encoder = Encoder()  # or Encoder(model_name=..., device=...)
    retriever = get_retriever(chunks, encoder, top_k=3)
    chain = get_prompt() | get_llm() | StrOutputParser()
    context = retrieve_context(
            question, retriever=retriever,
        )
    response = chain.invoke({"context": context, "question": question})
    return response
