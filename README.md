
<img src="docs/images/docuSun_demo_video_header.png" width="1000" />


# DocuSun - Local Document Assistant

DocuSun is a local-first AI document assistant for indexing files and answering questions from their content.


# Demo
[[Watch the demo of current version 0.2.0 on Youtube]](https://youtu.be/AxQKGX9mEa4)


## Platform Support

- **Supported now:** Linux only  


## Real-World Need

DocuSun is built to solve practical problems:

- Limited internet connectivity in many regions  
- Service interruptions during outages or disasters  
- Privacy concerns when documents are sensitive (academic, legal, medical, governmental)  
- Difficulty handling large files in cloud-only assistants  
- Need for multilingual accessibility  



## Project Goal

Build a local-first AI assistant that can:

- Ingest and process documents  
- Retrieve relevant context  
- Generate accurate, context-aware answers  
- Keep user data under local control  
- Evolve from CLI to full web experience  



## Current Version

**Current version:** v0.2.0 (Core engine + CLI workflow)

### What is available now:

- Document discovery and loading  
- Cleaning and chunking pipeline  
- Embedding and vector indexing  
- Retrieval and Q&A via CLI


## Language Support

- **Supported now:** English only



## Required Dependencies

Core stack includes:

- Python 3.10+  
- langchain, langchain-community  
- sentence-transformers  
- chromadb  
- rank_bm25  
- pymupdf  
- jsonargparse  
- rich  
- loguru  
- python-dotenv  
- streamlit (for future UI)  
- langchain-ollama (local LLM inference)  
- langchain-google-genai (Gemini API inference)  



## Installation (Terminal)

```bash
git clone https://github.com/HIS-MOHAMMED/docuSun.git
cd docuSun

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```



## Environment Setup

Create a `.env` file in the project root. Local is the default provider. Set `DOCUSUN_PROVIDER=api` to use Gemini via Google AI Studio.

```env
# Provider selection (default: local)
DOCUSUN_PROVIDER=local

# LLM model name (local or Gemini)
DOCUSUN_LLM_MODEL= # Local example: llama3.1:8b-instruct-q4_K_M | API example: gemini-2.5-flash

# Local LLM configuration (Ollama) - required when DOCUSUN_PROVIDER=local
DOCUSUN_OLLAMA_BASE_URL= # Your_ollama_base_url (e.g., http://localhost:11434)

# Gemini API key - required when DOCUSUN_PROVIDER=api
GOOGLE_API_KEY= # Your_Google_AI_Studio_API_key

# Multi-query retrieval toggle (default: disabled)
DOCUSUN_ENABLE_MULTI_QUERY=false

# Hybrid search toggle (default: disabled)
DOCUSUN_ENABLE_HYBRID_SEARCH=false

# Hybrid retriever tuning (optional)
DOCUSUN_HYBRID_VECTOR_WEIGHT=0.5
DOCUSUN_HYBRID_KEYWORD_WEIGHT=0.5
DOCUSUN_HYBRID_BM25_K=3

# Cohere API key - required when multi-query retrieval is enabled
COHERE_API_KEY= # Your_Cohere_API_key

# Cohere model used for multi-query generation (optional)
DOCUSUN_MULTI_QUERY_MODEL=command-r

# Embedding model name (local or Gemini)
DOCUSUN_EMBEDDING_MODEL= # Local example: google/embeddinggemma-300m | API example: text-embedding-004

# Used only for local embeddings
DOCUSUN_EMBEDDING_DEVICE= # Your_device (e.g., cpu, cuda, mps)

# Tokenizer model used for chunking (Hugging Face model name)
DOCUSUN_TOKENIZER_MODEL= # Your_tokenizer_model (e.g., gpt2)
```



## Quick CLI Usage

### Index your documents:
```bash
docusun index --data_path data --chunk_size 400 --top_k 3
```

### Ask a question:
```bash
docusun query --question "type_your_question_here" --top_k 3
```



## Roadmap Snapshot

- **v0.1.0:** Core ingestion and retrieval pipeline  
- **v0.2.0:** CLI interface  
- **v0.3.0:** Web interface  
- **v0.4.0:** Advanced retrieval (hybrid search, reranking, query expansion)  
- **v1.0.0:** API hardening and production release  
