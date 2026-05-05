# Embedding Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a script that reads `notebooks/chapter-2/llm.txt`, chunks it into 500-char pieces with 50-char overlap, embeds each chunk via OpenAI `text-embedding-3-small`, and stores text + metadata + embeddings to `al://mahmoudkamal01099s-organization/my_table` on Activeloop DeepLake.

**Architecture:** A single `index.py` script with four pure functions (`load_and_chunk`, `embed_batch`, `store_to_deeplake`, `main`) and a test file covering the chunker. Credentials are loaded from `.env` via `python-dotenv`. Chunks are embedded in batches of 200 to minimize API calls (~9 total for this file).

**Tech Stack:** Python 3.11+, `openai`, `deeplake`, `python-dotenv`, `pytest`, `.venv` managed with `uv`

---

## Task 1: Verify Dependencies Are Installed

**Step 1:** `source .venv/bin/activate && python -c "import openai, deeplake, dotenv; print('all good')"`
**Step 2:** If any fail → `uv add openai python-dotenv`
**Step 3:** `git add pyproject.toml uv.lock && git commit -m "chore: ensure dependencies present"`

---

## Task 2: Write Tests for `load_and_chunk`

**Files:**
- Create: `tests/test_chunker.py`

```python
# tests/test_chunker.py
import pytest
from index import load_and_chunk

def test_chunks_are_not_empty(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("A" * 1200)
    chunks = load_and_chunk(str(f), chunk_size=500, overlap=50)
    assert len(chunks) > 0

def test_chunk_size_respected(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("A" * 1200)
    chunks = load_and_chunk(str(f), chunk_size=500, overlap=50)
    for text, _ in chunks:
        assert len(text) <= 500

def test_chunk_index_sequential(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("B" * 1000)
    chunks = load_and_chunk(str(f), chunk_size=500, overlap=50)
    indices = [idx for _, idx in chunks]
    assert indices == list(range(len(chunks)))

def test_overlap_produces_shared_content(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("X" * 600)
    chunks = load_and_chunk(str(f), chunk_size=500, overlap=50)
    assert len(chunks) == 2
    assert chunks[0][0][-50:] == chunks[1][0][:50]

def test_empty_file_raises(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_and_chunk(str(f), chunk_size=500, overlap=50)
```

Run: `pytest tests/test_chunker.py -v` → Expected: FAIL (not implemented yet)

`git add tests/test_chunker.py && git commit -m "test: add chunker tests for load_and_chunk"`

---

## Task 3: Implement `load_and_chunk` in `index.py`

**Files:**
- Modify: `index.py` (replace entire file)

```python
# index.py
import os
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()


def load_and_chunk(
    file_path: str, chunk_size: int = 500, overlap: int = 50
) -> List[Tuple[str, int]]:
    """Read a text file and split into fixed-size chunks with overlap."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        raise ValueError(f"File '{file_path}' is empty or contains only whitespace.")

    chunks = []
    step = chunk_size - overlap
    start = 0
    index = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append((chunk, index))
        start += step
        index += 1

    return chunks
```

Run: `pytest tests/test_chunker.py -v` → Expected: `5 passed`

`git add index.py && git commit -m "feat: implement load_and_chunk with fixed-size chunking and overlap"`

---

## Task 4: Implement `embed_batch` and `store_to_deeplake`

**Files:**
- Modify: `index.py` (add imports + append two functions)

Add at the top:
```python
import numpy as np
from openai import OpenAI
```

Append after `load_and_chunk`:
```python
def embed_batch(client: OpenAI, texts: List[str]) -> List[List[float]]:
    """Call OpenAI embeddings API for a batch of texts. Returns 1536-dim vectors."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


def store_to_deeplake(ds, texts: List[str], metadatas: List[dict], embeddings: List[List[float]]) -> None:
    """Append one batch of chunks to all three DeepLake tensors."""
    with ds:
        for text, meta, emb in zip(texts, metadatas, embeddings):
            ds.text.append(text)
            ds.metadata.append(meta)
            ds.embedding.append(np.array(emb, dtype="float32"))
```

Verify: `python -c "from index import load_and_chunk, embed_batch, store_to_deeplake; print('OK')"`

Run: `pytest tests/test_chunker.py -v` → Expected: `5 passed`

`git add index.py && git commit -m "feat: add embed_batch and store_to_deeplake functions"`

---

## Task 5: Implement `main()` Orchestrator

**Files:**
- Modify: `index.py` (add `deeplake` import + append constants, `main()`, `__main__` block)

Add import at top: `import deeplake`

Append to `index.py`:
```python
DATASET_PATH = "al://mahmoudkamal01099s-organization/my_table"
SOURCE_FILE   = "notebooks/chapter-2/llm.txt"
CHUNK_SIZE    = 500
OVERLAP       = 50
BATCH_SIZE    = 200


def main() -> None:
    openai_key = os.getenv("OPENAI_API_KEY")
    activeloop_token = os.getenv("ACTIVELOOP_TOKEN")
    if not openai_key:
        raise EnvironmentError("OPENAI_API_KEY not found in environment / .env")
    if not activeloop_token:
        raise EnvironmentError("ACTIVELOOP_TOKEN not found in environment / .env")

    client = OpenAI(api_key=openai_key)

    print(f"Loading and chunking '{SOURCE_FILE}' ...")
    chunks = load_and_chunk(SOURCE_FILE, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    print(f"  -> {len(chunks)} chunks created")

    print(f"Connecting to DeepLake: {DATASET_PATH}")
    ds = deeplake.empty(DATASET_PATH, overwrite=True, token=activeloop_token)
    ds.create_tensor("text",      htype="text")
    ds.create_tensor("metadata",  htype="json")
    ds.create_tensor("embedding", htype="embedding", dtype="float32",
                     sample_compression=None)

    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_num, batch_start in enumerate(range(0, len(chunks), BATCH_SIZE), start=1):
        batch     = chunks[batch_start: batch_start + BATCH_SIZE]
        texts     = [c[0] for c in batch]
        metadatas = [{"source": SOURCE_FILE, "chunk_index": c[1]} for c in batch]

        print(f"  Embedding batch {batch_num}/{total_batches} ({len(texts)} chunks) ...",
              end=" ", flush=True)
        try:
            embeddings = embed_batch(client, texts)
        except Exception as e:
            print(f"\nBatch {batch_num} failed: {e}")
            raise

        store_to_deeplake(ds, texts, metadatas, embeddings)
        print(f"done  ({batch_start + len(texts)}/{len(chunks)} stored)")

    print(f"\nDone! {len(chunks)} chunks embedded and stored to {DATASET_PATH}")


if __name__ == "__main__":
    main()
```

Verify: `python -c "from index import main; print('main OK')"`

`git add index.py && git commit -m "feat: add main() orchestrator for full embedding pipeline"`

---

## Task 6: Run the Script End-to-End

**Step 1:** `source .venv/bin/activate && python index.py`

Expected:
```
Loading and chunking 'notebooks/chapter-2/llm.txt' ...
  -> ~1650 chunks created
Connecting to DeepLake: al://mahmoudkamal01099s-organization/my_table
  Embedding batch 1/9 (200 chunks) ... done  (200/1650 stored)
  ...
Done! 1650 chunks embedded and stored to al://mahmoudkamal01099s-organization/my_table
```

**Step 2:** Verify at https://deeplake.ai/mahmoudkamal01099s-organization/workspace/default/table/my_table
- Rows present, `text`/`metadata`/`embedding` columns populated
- Embedding shape: `(1536,)`

**Step 3:** `git add . && git commit -m "feat: complete embedding pipeline — llm.txt embedded to DeepLake"`

---

## File Summary

| File | Action |
|---|---|
| `index.py` | Full rewrite — embedding pipeline |
| `tests/test_chunker.py` | New — unit tests for `load_and_chunk` |
