# VisDrone RAG Pipeline — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a 3-pipeline RAG system for drone/computer vision content using LlamaIndex + DeepLake in an independent folder.

**Architecture:** Three sequential pipelines — data collection → vector store creation → query engines. Uses LlamaIndex's `DeepLakeVectorStore` integration with local dataset path to avoid overriding existing cloud tables. Both VectorStoreIndex (similarity) and TreeIndex (hierarchical) query engines.

**Tech Stack:** LlamaIndex (core, embeddings-openai, llms-openai, vector-stores-deeplake), BeautifulSoup4, requests, python-dotenv

**Design Doc:** `docs/plans/2026-06-10-visdrone-rag-design.md`

---

### Task 1: Create folder structure and config

**Files:**
- Create: `visdrone_rag/__init__.py`
- Create: `visdrone_rag/config.py`
- Modify: `.gitignore` (add data/store/index dirs)

**Step 1: Create folder and empty `__init__.py`**

**Step 2: Create `visdrone_rag/config.py`** — loads `../.env`, exports `OPENAI_API_KEY`, `ACTIVELOOP_TOKEN`, `DATA_DIR`, `DEEPLAKE_DATASET_PATH`, `TREE_INDEX_DIR`. Sets `os.environ` for LlamaIndex auto-detection.

**Step 3: Update `.gitignore`** — append `visdrone_rag/data/`, `visdrone_rag/deeplake_store/`, `visdrone_rag/indexes/`

**Step 4: Commit** — `feat(visdrone): add folder structure and config`

---

### Task 2: Pipeline 1 — Data Collection & Cleaning

**Files:**
- Create: `visdrone_rag/pipeline_1_collect_data.py`

**Step 1: Create the script** with:
- All 24 URLs from the user's list
- `clean_text()` — regex to remove `[n]` references and punctuation (except periods)
- `fetch_and_clean()` — requests + BeautifulSoup, `mw-parser-output` → `content` fallback, section removal with while loop for nested sections
- `main()` — iterate URLs, save to `./data/{article_name}.txt`, skip failures

**Step 2: Run** — `uv run python visdrone_rag/pipeline_1_collect_data.py`

**Step 3: Commit** — `feat(visdrone): add pipeline 1 - data collection and cleaning`

---

### Task 3: Pipeline 2 — DeepLake Vector Store via LlamaIndex

**Files:**
- Create: `visdrone_rag/pipeline_2_vector_store.py`

**Step 1: Create the script** with:
- Import config, configure `Settings.embed_model` = `OpenAIEmbedding(model="text-embedding-3-small")`, `Settings.llm` = `OpenAI(model="gpt-4o-mini")`
- `SimpleDirectoryReader(input_dir=DATA_DIR, required_exts=[".txt"]).load_data()`
- `DeepLakeVectorStore(dataset_path=DEEPLAKE_DATASET_PATH, overwrite=True)` — LOCAL path, no cloud conflicts
- `VectorStoreIndex.from_documents(documents, storage_context, show_progress=True)`

**Step 2: Run** — `uv run python visdrone_rag/pipeline_2_vector_store.py`

**Step 3: Commit** — `feat(visdrone): add pipeline 2 - DeepLake vector store via LlamaIndex`

---

### Task 4: Pipeline 3 — VectorStoreIndex & TreeIndex Query Engines

**Files:**
- Create: `visdrone_rag/pipeline_3_rag_query.py`

**Step 1: Create the script** with:
- `build_vector_query_engine()` — reload `DeepLakeVectorStore`, `VectorStoreIndex.from_vector_store()`, `as_query_engine(similarity_top_k=5, response_mode="compact")`
- `build_tree_query_engine()` — check if `TREE_INDEX_DIR` exists → load from storage, else build `TreeIndex.from_documents()` and persist. `as_query_engine(response_mode="tree_summarize")`
- `main()` — interactive CLI: choose engine [1/2/q] → enter question → display response

**Step 2: Run** — `uv run python visdrone_rag/pipeline_3_rag_query.py`

**Step 3: Commit** — `feat(visdrone): add pipeline 3 - vector store and tree index query engines`

---

### Task 5: README and final commit

**Files:**
- Create: `visdrone_rag/README.md`

**Step 1: Create README** with setup instructions, usage commands for all 3 pipelines, and architecture overview.

**Step 2: Commit** — `docs(visdrone): add README with usage instructions`
