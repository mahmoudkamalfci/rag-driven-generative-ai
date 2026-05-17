# Vector Store Ingestion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a script to chunk `llm.txt`, generate OpenAI embeddings, and ingest them into DeepLake.

**Architecture:** A standalone Python script `data_vector_store.py` that uses batching to process text chunks, embedding them via OpenAI and storing in a DeepLake table.

**Tech Stack:** Python, OpenAI SDK, DeepLake SDK.

---

### Task 1: Add OpenAI Dependency

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Step 1: Write the failing test**
N/A - Dependency addition.

**Step 2: Run test to verify it fails**
N/A

**Step 3: Write minimal implementation (Install openai)**
Run: `uv add openai`
Expected: OpenAI is added to the project dependencies.

**Step 4: Run test to verify it passes**
Run: `uv pip list | grep openai`
Expected: OpenAI is listed.

**Step 5: Commit**
```bash
git add pyproject.toml uv.lock
git commit -m "chore: add openai dependency"
```

### Task 2: Implement and Test Text Chunking

**Files:**
- Create: `data_vector_store.py`
- Create: `test_data_vector_store.py`

**Step 1: Write the failing test**
Create `test_data_vector_store.py`:
```python
from data_vector_store import chunk_text

def test_chunk_text():
    text = "A" * 2500
    chunks = chunk_text(text, chunk_size=1000)
    assert len(chunks) == 3
    assert len(chunks[0]) == 1000
    assert len(chunks[2]) == 500
```

**Step 2: Run test to verify it fails**
Run: `pytest test_data_vector_store.py -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

**Step 3: Write minimal implementation**
Create `data_vector_store.py`:
```python
def chunk_text(text: str, chunk_size: int = 1000) -> list[str]:
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
```

**Step 4: Run test to verify it passes**
Run: `pytest test_data_vector_store.py -v`
Expected: PASS

**Step 5: Commit**
```bash
git add data_vector_store.py test_data_vector_store.py
git commit -m "feat: implement text chunking"
```

### Task 3: Implement Embedding and Ingestion Logic

**Files:**
- Modify: `data_vector_store.py`

**Step 1: Write minimal implementation**
Modify `data_vector_store.py` to add the main ingestion logic:
```python
import os
import openai
from dotenv import load_dotenv
from deeplake import Client
from data_vector_store import chunk_text

def main():
    load_dotenv()
    
    openai.api_key = os.getenv("OPENAI_API_KEY")
    activeloop_token = os.getenv("ACTIVELOOP_TOKEN")
    
    if not openai.api_key or not activeloop_token:
        print("Missing API keys")
        return
        
    client = Client(token=activeloop_token, workspace_id="first")
    
    # Drop table if exists for a fresh start
    try:
        client.drop_table("llm_embeddings")
        print("Dropped existing table.")
    except Exception:
        pass
        
    with open("llm.txt", "r") as f:
        text = f.read()
        
    chunks = chunk_text(text, 1000)
    batch_size = 200
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}...")
        
        response = openai.embeddings.create(input=batch, model="text-embedding-3-small")
        embeddings = [data.embedding for data in response.data]
        
        client.ingest("llm_embeddings", {"text": batch, "embedding": embeddings})
        
    print("Creating index...")
    client.create_index("llm_embeddings", "embedding")
    client.create_index("llm_embeddings", "text")
    print("Done!")

if __name__ == "__main__":
    main()
```

**Step 2: Run script to verify it works**
Run: `python data_vector_store.py`
Expected: Output showing batches processing, index creation, and "Done!"

**Step 3: Commit**
```bash
git add data_vector_store.py
git commit -m "feat: implement batch vector ingestion"
```
