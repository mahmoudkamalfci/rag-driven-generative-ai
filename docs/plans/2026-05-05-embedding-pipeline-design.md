# Embedding Pipeline Design

**Date:** 2026-05-05  
**Status:** Approved  
**Topic:** Embed `llm.txt` using OpenAI and store to Activeloop DeepLake

---

## Goal

Create a simple Python script (`index.py`) that:
1. Reads `notebooks/chapter-2/llm.txt`
2. Splits it into fixed-size chunks
3. Embeds each chunk using OpenAI's `text-embedding-3-small`
4. Stores chunks, metadata, and embeddings into a DeepLake cloud dataset

---

## Configuration

| Parameter | Value |
|---|---|
| Source file | `notebooks/chapter-2/llm.txt` |
| Chunk size | 500 characters |
| Chunk overlap | 50 characters |
| Batch size | 200 chunks per OpenAI API call |
| Embedding model | `text-embedding-3-small` |
| Embedding dimensions | 1536 |
| DeepLake dataset | `al://mahmoudkamal01099s-organization/my_table` |
| Credentials | `OPENAI_API_KEY`, `ACTIVELOOP_TOKEN` loaded from `.env` |

---

## Architecture & Data Flow

```
llm.txt (842KB)
    │
    ▼
[1. Load & Chunk]
    • Read full file content
    • Split into 500-char chunks with 50-char overlap
    • Each chunk gets metadata: { "source": "llm.txt", "chunk_index": N }
    │
    ▼
[2. Batch Embed]
    • Group chunks into batches of 200
    • Call openai.embeddings.create(model="text-embedding-3-small", input=batch)
    • Returns 1536-dim float32 vectors
    │
    ▼
[3. Store to DeepLake]
    • Connect to al://mahmoudkamal01099s-organization/my_table
    • Append to 3 tensors per batch:
        - text       → raw chunk string
        - metadata   → { source, chunk_index }
        - embedding  → 1536-dim float32 vector
    │
    ▼
[4. Progress Reporting]
    • Print after each batch: "Batch N/Total — X/Y chunks stored"
```

**Estimated scale:** ~1,650 chunks → ~9 batches of 200

---

## Components

| Function | Responsibility |
|---|---|
| `load_and_chunk(file_path, chunk_size, overlap)` | Reads file, returns list of `(chunk_text, chunk_index)` tuples |
| `embed_batch(client, texts)` | Calls OpenAI API for one batch, returns list of float32 vectors |
| `store_to_deeplake(ds, texts, metadatas, embeddings)` | Appends one batch to all 3 DeepLake tensors |
| `main()` | Orchestrates: load env → init clients → chunk → batch loop → report |

---

## Error Handling

- **Missing `.env` keys** → raise early with a clear message (fail fast before any API calls)
- **OpenAI API error** → catch `openai.APIError`, print batch info, re-raise
- **DeepLake connection failure** → let SDK error bubble naturally
- **Empty file** → guard check before chunking, raise with clear message
- **No retry logic** — simple script; stops cleanly on failure

---

## DeepLake Tensor Schema

| Tensor | htype | dtype |
|---|---|---|
| `text` | `text` | default (str) |
| `metadata` | `json` | default |
| `embedding` | `embedding` | `float32`, no compression, shape `(1536,)` |

---

## Approach Considered

| Option | Notes |
|---|---|
| Sequential (one API call/chunk) | Simple but ~1,650 API calls — too slow |
| **Batched (chosen)** | ~9 API calls, fast, no extra dependencies |
| LangChain abstraction | Less code, but heavy dependency not needed |
