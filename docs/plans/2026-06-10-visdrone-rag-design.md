# VisDrone RAG Pipeline — Design Document

## Purpose

Build a 3-pipeline RAG system focused on drone/computer vision content using LlamaIndex with DeepLake as the vector store. This lives in an independent `visdrone_rag/` folder within the existing `rag-driven-generative-ai` project.

## Constraints

- Must NOT override existing DeepLake cloud tables (`llm_embeddings`, `my_first_table` in workspace `first`)
- Must reuse existing `.env` credentials (`OPENAI_API_KEY`, `ACTIVELOOP_TOKEN`)
- Must match existing project patterns: `text-embedding-3-small` for embeddings, `gpt-4o-mini` for LLM
- Must use LlamaIndex's built-in `DeepLakeVectorStore` integration (Option A from brainstorming)
- All new code lives inside `visdrone_rag/` — independent from root-level scripts

## Success Criteria

1. Pipeline 1 fetches all 24 URLs and saves cleaned text files
2. Pipeline 2 creates a local DeepLake vector store (no cloud table conflicts) with LlamaIndex embeddings
3. Pipeline 3 provides both VectorStoreIndex and TreeIndex query engines with an interactive CLI

## Architecture

```
visdrone_rag/
├── data/                          # Scraped & cleaned text files (gitignored)
├── deeplake_store/                # Local DeepLake dataset (gitignored)
├── indexes/                       # Persisted tree index (gitignored)
│   └── tree_index/
├── config.py                      # Shared env setup — loads ../.env
├── pipeline_1_collect_data.py     # Fetch, clean, save to ./data/
├── pipeline_2_vector_store.py     # Create DeepLake vector store via LlamaIndex
├── pipeline_3_rag_query.py        # VectorStoreIndex + TreeIndex query engines
└── README.md                      # Usage instructions
```

## Data Flow

```
URLs (24) → BeautifulSoup scraping → clean_text() → .txt files in data/
                                                          ↓
                                          SimpleDirectoryReader loads documents
                                                    ↓                ↓
                                     DeepLakeVectorStore          TreeIndex
                                     (local dataset)             (persisted to disk)
                                           ↓                        ↓
                                   VectorStoreIndex             TreeIndex
                                     query engine              query engine
                                           ↓                        ↓
                                        Interactive CLI (choose engine, ask questions)
```

## Pipeline 1: Data Collection & Cleaning

- Uses the user's exact cleaning logic from the request
- `mw-parser-output` div → falls back to `content` id div
- Removes sections: References, Bibliography, External links, See also, Notes (with nested removal via while loop)
- Regex: strips `[n]` references and punctuation except periods
- Each URL saved as `{last_path_segment}.txt` in `./data/`
- Graceful failure: returns None for unreachable URLs

## Pipeline 2: DeepLake Vector Store

- `SimpleDirectoryReader` loads all `.txt` files from `./data/`
- `DeepLakeVectorStore(dataset_path="./visdrone_rag/deeplake_store", overwrite=True)` — local path, no cloud conflicts
- `VectorStoreIndex.from_documents()` with `StorageContext` — LlamaIndex handles chunking + embedding automatically
- Uses `OpenAIEmbedding(model="text-embedding-3-small")`

## Pipeline 3: RAG Query Engines

- **Vector Store Index**: Reloads from local DeepLake store via `VectorStoreIndex.from_vector_store()`, creates query engine with `similarity_top_k=5`, `response_mode="compact"`
- **Tree Index**: Builds `TreeIndex.from_documents()`, persists to `./indexes/tree_index/`, creates query engine with `response_mode="tree_summarize"`. Loads from disk on subsequent runs.
- Interactive CLI: choose engine → enter question → see response

## Dependencies (already installed)

- `llama-index-core==0.14.22`
- `llama-index-vector-stores-deeplake==0.5.0`
- `llama-index-embeddings-openai==0.6.0`
- `llama-index-llms-openai==0.7.9`
- Existing: `beautifulsoup4`, `requests`, `python-dotenv`, `openai`, `deeplake`
