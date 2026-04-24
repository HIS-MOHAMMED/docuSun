
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
- langchain-openai / API-compatible model access  



## Installation (Terminal)

```bash
git clone <your-repo-url>
cd docuSun

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```



## Environment Setup

Create a `.env` file in the project root (if using API-based models):

```env
DocuSun_GITHUB_TOKEN=your_token_here
DOCUSUN_EMBEDDING_PROVIDER=api
DOCUSUN_EMBEDDING_MODEL=text-embedding-3-small
DOCUSUN_EMBEDDING_DEVICE=cpu
DOCUSUN_TOKENIZER_MODEL=gpt2
```

For local embeddings, switch provider to `local` and choose a local embedding model name.



## Quick CLI Usage

### Index your documents:
```bash
docusun index --data_path data --chunk_size 400 --top_k 3
```

### Ask a question:
```bash
docusun query --question "What are the main findings?" --top_k 3
```



## Roadmap Snapshot

- **v0.1.0:** Core ingestion and retrieval pipeline  
- **v0.2.0:** CLI interface  
- **v0.3.0:** Web interface  
- **v0.4.0:** Advanced retrieval (hybrid search, reranking, query expansion)  
- **v1.0.0:** API hardening and production release  
