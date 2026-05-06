
<img src="docs/images/docuSun_demo_video_header.png" width="1000" />


# DocuSun - Local Document Assistant

DocuSun is local AI document assistant for indexing local files and answering questions from their content.


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

Create a `.env` file in the project root. `DOCUSUN_LLM_MODEL`, `DOCUSUN_OLLAMA_BASE_URL`, `DOCUSUN_EMBEDDING_MODEL`, and `DOCUSUN_TOKENIZER_MODEL` are required:

```env
DOCUSUN_LLM_MODEL= # Your_installed_local_llm (e.g., llama3.1:8b-instruct-q4_K_M)
DOCUSUN_OLLAMA_BASE_URL= # Your_ollama_base_url (e.g., http://localhost:11434)
DOCUSUN_EMBEDDING_MODEL= # Your_local_embedding_model (e.g., google/embeddinggemma-300m)
DOCUSUN_EMBEDDING_DEVICE= # Your_device (e.g., cpu, cuda, mps)
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
