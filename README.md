
<img src="docs/images/docuSun v0.1_ Local search and indexing.png" width="600" />
# docuSun

DocuSun is a local Retrieval‑Augmented Generation (RAG) engine for privately indexing and querying documents using open‑source embedding and language models.

## Current features
- Recursive file discovery for common document types (.pdf, .txt, .docx)
- PDF loader that extracts page‑level text into RawDoc objects
- Text cleaners (whitespace normalization, paragraph grouping)
- Recursive text splitter with configurable chunk size and overlap
- Embedding wrapper (default: HuggingFace BGE) and embedding utilities
- Chroma vector store integration for persisting embeddings
- Retrieval module to fetch and rerank top‑k relevant chunks

## Planned / future features
- Integrate a local LLM to generate answers from retrieved context and add citation support
- CLI and lightweight web API (FastAPI) for ingestion and querying
- Web interface (FastAPI + minimal browser UI for upload, status, conversational querying and session history)
- Additional embedding backends, hybrid retrieval (BM25 + vector), and improved reranking
- Vector store lifecycle tools (backup/restore, update/delete) and evaluation scripts
