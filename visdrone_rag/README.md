# VisDrone RAG Pipeline

RAG system for drone/computer vision content using LlamaIndex + DeepLake.

## Setup

Dependencies are already in the project's `pyproject.toml` (and managed via `uv`). Ensure the root `.env` file has:
- `OPENAI_API_KEY`
- `ACTIVELOOP_TOKEN`

## Usage

Run the pipelines in order from the project root:

### 1. Collect Data
```bash
uv run python visdrone_rag/pipeline_1_collect_data.py
```
This fetches 24 target URLs (Wikipedia, papers, datasets related to VisDrone/object detection), cleans the HTML, and saves `.txt` files to `./visdrone_rag/data/`.

### 2. Create Vector Store
```bash
uv run python visdrone_rag/pipeline_2_vector_store.py
```
This loads the `.txt` files into LlamaIndex's `DeepLakeVectorStore`. It creates a local dataset at `./visdrone_rag/deeplake_store/` so it won't conflict with existing cloud tables.

### 3. Query
```bash
uv run python visdrone_rag/pipeline_3_rag_query.py
```
This script opens an interactive CLI with two query engines:
- **Vector Store Index** (cosine similarity search, fast and compact)
- **Tree Index** (hierarchical summarization, parses relationships across chunks)
