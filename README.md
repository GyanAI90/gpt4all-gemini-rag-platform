# GPT4All + Gemini RAG Intelligence Platform

> A dual-model Retrieval-Augmented Generation (RAG) platform combining local GPT4All inference, Google Gemini, Sentence Transformers embeddings, ChromaDB vector search, source grounding, evaluation, and a Streamlit interface.

## Overview

This project implements an end-to-end document intelligence pipeline:

```text
PDF Documents
     |
     v
Document Loader
     |
     v
Chunking
     |
     v
Sentence Transformers
     |
     v
ChromaDB Vector Store
     |
     v
Semantic Retriever
     |
     +----------------------+
     |                      |
     v                      v
GPT4All / Qwen2-1.5B     Gemini
     |                      |
     +----------+-----------+
                |
                v
        Answer + Sources
                |
                v
           Evaluation
                |
                v
          Streamlit UI
```

The platform supports three inference modes:

- **GPT4All** — local Qwen2-1.5B-Instruct inference
- **Gemini** — cloud LLM inference
- **Compare Both** — generates responses using the same retrieved context and compares them

## Key Features

- PDF document ingestion
- Page-level metadata preservation
- Configurable recursive text chunking
- `sentence-transformers/all-MiniLM-L6-v2` embeddings
- 384-dimensional semantic vectors
- Persistent ChromaDB vector store
- Top-K semantic retrieval
- Local GPT4All inference
- Gemini cloud inference
- Dual-model comparison
- Source metadata and retrieved chunk inspection
- Retrieval relevance evaluation
- Answer groundedness evaluation
- Source coverage evaluation
- Pipeline timing metrics
- Streamlit chat interface

## Development Validation

The current development dataset contains:

```text
Documents:              3
Total pages:          254
Total chunks:        1545
Chunk size:           1000
Chunk overlap:        200
Embedding dimension:  384
Vector records:      1545
Retriever Top-K:       5
```

Validated documents:

```text
caterpillar_2025_annual_report.pdf
osha_safety_manual.pdf
rag_llm_development_guide.pdf
```

Example query:

> What are Caterpillar's major business risks?

The pipeline successfully retrieves relevant sections from the indexed documents and generates responses through both GPT4All and Gemini.

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python |
| UI | Streamlit |
| PDF Processing | PyPDF |
| Chunking | LangChain Text Splitters |
| Embeddings | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| Local LLM | GPT4All |
| Local Model | Qwen2-1.5B-Instruct |
| Model Format | GGUF |
| Cloud LLM | Google Gemini |
| Configuration | python-dotenv |

## Project Structure

```text
gpt4all-gemini-rag-platform/
|
+-- app.py
+-- README.md
+-- requirements.txt
+-- .gitignore
+-- .env.example
|
+-- src/
|   +-- document_loader.py
|   +-- chunking.py
|   +-- embeddings.py
|   +-- vector_store.py
|   +-- retriever.py
|   +-- gpt4all_llm.py
|   +-- gemini_llm.py
|   +-- rag_pipeline.py
|   +-- dual_rag_pipeline.py
|   +-- evaluator.py
|
+-- data/
|   +-- pdfs/
|
+-- .streamlit/
    +-- config.toml
```

Generated directories such as `venv/` and `chroma_db/` should not be committed.

## RAG Pipeline

### 1. Document Loading

`document_loader.py` loads PDFs and preserves document/page information.

### 2. Chunking

`chunking.py` converts page documents into overlapping text chunks.

Current development configuration:

```text
Chunk size:    1000 characters
Chunk overlap: 200 characters
```

### 3. Embeddings

`embeddings.py` uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The resulting embeddings have 384 dimensions.

### 4. Vector Database

`vector_store.py` stores chunk text, embeddings, and metadata in ChromaDB.

### 5. Retrieval

`retriever.py` performs semantic similarity search.

Current configuration:

```text
Top-K = 5
```

Retrieved results retain:

- document name
- page number
- chunk ID
- similarity/distance
- chunk content

### 6. Generation

`dual_rag_pipeline.py` passes the retrieved context to:

```text
GPT4All / Qwen2-1.5B
Gemini
```

The same retrieved evidence can therefore be used for a controlled local-vs-cloud comparison.

## Evaluation

`evaluator.py` provides lightweight RAG quality and performance metrics.

### Retrieval Relevance

Measures keyword-level relevance between the question and retrieved documents.

### Groundedness

Estimates how strongly generated content is supported by retrieved context.

### Source Coverage

Measures how much answer content can be associated with retrieved sources.

### Performance

The pipeline records timing for retrieval and model generation.

A development evaluation produced approximately:

```text
Retrieval latency:    0.44 sec
Retrieved documents:      5

GPT4All groundedness: 0.708
Gemini groundedness:  0.814
```

These are development-test measurements, not universal benchmarks.

## Installation

### Requirements

Recommended development environment:

- Python 3.10+
- 16 GB RAM
- Internet access for initial package/model downloads
- Gemini API key for cloud inference

GPU is optional for the local Qwen2-1.5B configuration.

### Clone

```bash
git clone https://github.com/YOUR_USERNAME/gpt4all-gemini-rag-platform.git
cd gpt4all-gemini-rag-platform
```

### Virtual Environment

Windows:

```bat
python -m venv venv
venv\Scripts\activate
```

### Dependencies

```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

Copy:

```text
.env.example
```

to:

```text
.env
```

Configure your Gemini credentials:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=your_configured_gemini_model
```

Never commit `.env`.

## GPT4All Model

The project uses:

```text
Qwen2-1.5B-Instruct
```

in GGUF format.

Model files are intentionally excluded from Git because of their size. Download the model separately and configure the path used by `src/gpt4all_llm.py`.

## Build the Vector Database

Place source PDFs in:

```text
data/pdfs/
```

Run the pipeline components individually:

```bat
python src\document_loader.py
python src\chunking.py
python src\embeddings.py
python src\vector_store.py
python src\retriever.py
```

## Test the RAG Components

Local GPT4All RAG:

```bat
python src\rag_pipeline.py
```

Gemini:

```bat
python src\gemini_llm.py
```

Dual-model RAG:

```bat
python src\dual_rag_pipeline.py
```

Evaluation:

```bat
python src\evaluator.py
```

## Run the Streamlit Application

```bat
python -m streamlit run app.py
```

If Streamlit's file watcher causes optional `torchvision` import errors in the local environment:

```bat
python -m streamlit run app.py --server.fileWatcherType none
```

For a permanent setting:

```toml
[server]
fileWatcherType = "none"
```

Save that as:

```text
.streamlit/config.toml
```

## Example Queries

```text
What are Caterpillar's major business risks?

What environmental risks does Caterpillar identify?

What are the major supply chain risks?

What safety procedures are described in the OSHA manual?

Explain Retrieval Augmented Generation based on the provided documents.

Compare the risks identified across the available documents.
```

## Why These Components?

### ChromaDB

Provides a simple persistent vector database suitable for local RAG development.

### MiniLM

`all-MiniLM-L6-v2` provides a practical balance between embedding quality, memory usage, and CPU performance.

### GPT4All

Enables local LLM inference and reduces dependence on a cloud API for the local inference path.

### Gemini

Provides a second inference path for quality comparison and cloud-based generation.

## Privacy and Security

The GPT4All path can operate locally after model installation.

When Gemini mode is used, retrieved context is sent to the Gemini API. Do not send confidential data to a cloud model unless the workflow is approved for that data.

Keep credentials out of source control:

```text
.env
API keys
access tokens
private certificates
model files
```

## Limitations

- Semantic retrieval does not guarantee perfect evidence selection.
- Character-based chunking can split semantic structures.
- The included evaluation metrics are lightweight engineering metrics, not a replacement for comprehensive human evaluation.
- Qwen2-1.5B is intentionally small and may be weaker than larger models on complex reasoning.
- Citation quality depends on the generation and source-tracking implementation.
- The development vector database is generated locally and is intentionally excluded from Git.

## Roadmap

- [ ] Advanced relevance filtering
- [ ] Cross-encoder reranking
- [ ] Better context construction
- [ ] Robust source citation formatting
- [ ] Improved answer-not-found detection
- [ ] Automated retrieval evaluation dataset
- [ ] RAGAS-style evaluation
- [ ] Query rewriting
- [ ] Hybrid BM25 + vector retrieval
- [ ] Metadata filtering
- [ ] Streaming responses
- [ ] Document upload from UI
- [ ] Persistent conversation management
- [ ] Authentication and authorization
- [ ] Docker deployment
- [ ] Production vector database option
- [ ] Structured logging and observability

## Engineering Highlights

```text
✓ End-to-end RAG architecture
✓ Local LLM inference
✓ Cloud LLM integration
✓ Vector database integration
✓ Semantic embeddings
✓ PDF document processing
✓ Context-aware generation
✓ Source metadata tracking
✓ Retrieval evaluation
✓ Groundedness evaluation
✓ Model comparison
✓ Streamlit application
✓ Environment and secret management
```

## License

Choose a license appropriate for your intended use before publishing.

If you own the application code and want a permissive open-source license, MIT is a common option.

Third-party PDFs and model files may have separate licensing or redistribution restrictions.

## Author

**Gyan Singh**

AI / ML Engineering Project

## Project Status

**Functional Development Release**

The complete development workflow has been tested:

```text
PDF ingestion
    ↓
Chunking
    ↓
Embeddings
    ↓
ChromaDB
    ↓
Semantic Retrieval
    ↓
GPT4All + Qwen2-1.5B
    +
Gemini
    ↓
RAG Evaluation
    ↓
Streamlit UI
```

The project is ready for GitHub publication and continued development toward production-grade RAG evaluation and deployment.
