# Augmented Generation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a standalone Python script (`augmented_generation.py`) to run a stateless interactive Retrieval-Augmented Generation CLI using DeepLake and OpenAI.

**Architecture:** The script will load environment variables, connect to the DeepLake `llm_embeddings` table and OpenAI, and run an interactive loop. In each iteration, it takes user input, retrieves relevant context from DeepLake via OpenAI embeddings, and queries `gpt-4o-mini` with the augmented prompt.

**Tech Stack:** Python 3.12, `openai`, `deeplake`, `python-dotenv`.

---

### Task 1: Create Prompt Augmentation Utility

**Files:**
- Create: `test/test_augmented_generation.py`
- Create: `augmented_generation.py`

**Step 1: Write the failing test**

```python
from augmented_generation import augment_prompt

def test_augment_prompt():
    user_prompt = "What is Mars?"
    context = "Mars is a red planet."
    expected = "What is Mars?\n\nContext:\nMars is a red planet."
    assert augment_prompt(user_prompt, context) == expected
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_augmented_generation.py -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

**Step 3: Write minimal implementation**

In `augmented_generation.py`:
```python
def augment_prompt(user_prompt: str, context: str) -> str:
    return f"{user_prompt}\n\nContext:\n{context}"
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_augmented_generation.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add test/test_augmented_generation.py augmented_generation.py
git commit -m "feat: add prompt augmentation utility with test"
```

---

### Task 2: Implement Retrieval & LLM Generation Logic

**Files:**
- Modify: `augmented_generation.py`

**Step 1: Write the core logic**

Since this heavily relies on external APIs (OpenAI and DeepLake), we'll implement the logic directly rather than mocking it extensively.

```python
import os
import openai
from dotenv import load_dotenv
from deeplake import Client

# pyrefly: ignore [unexpected-keyword]
def setup_clients():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    activeloop_token = os.getenv("ACTIVELOOP_TOKEN")
    
    if not openai.api_key or not activeloop_token:
        print("Missing API keys in .env file.")
        return None, None
        
    client = Client(token=activeloop_token, workspace_id="first")
    return openai, client

def get_answer(openai_client, deeplake_client, query: str) -> str:
    # 1. Embed query
    response = openai_client.embeddings.create(input=[query], model="text-embedding-3-small")
    query_embedding = response.data[0].embedding
    
    # 2. Search DeepLake
    search_results = deeplake_client.search("llm_embeddings", embedding=query_embedding)
    
    # Check if we have results
    if not search_results['text']:
        return "No relevant context found."
        
    top_text = search_results['text'][0]
    top_score = search_results['score'][0]
    print(f"\n[Debug] Context Match Score: {top_score:.4f}\n")
    
    # 3. Augment Prompt
    augmented = augment_prompt(query, top_text)
    
    # 4. Ask LLM
    chat_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a space exploration expert. Use the provided context to answer the user's question in detail."},
            {"role": "user", "content": augmented}
        ],
        temperature=0.1
    )
    return chat_response.choices[0].message.content
```

**Step 2: Commit**

```bash
git add augmented_generation.py
git commit -m "feat: implement retrieval and LLM generation logic"
```

---

### Task 3: Implement Interactive Loop & Main Entry Point

**Files:**
- Modify: `augmented_generation.py`

**Step 1: Write the interactive loop**

```python
def main():
    openai_client, deeplake_client = setup_clients()
    if not openai_client or not deeplake_client:
        return
        
    print("Welcome to the Space Exploration RAG CLI!")
    print("Type 'quit' or 'exit' to stop.")
    
    while True:
        try:
            user_input = input("\nEnter your question: ").strip()
            if user_input.lower() in ['quit', 'exit']:
                break
                
            if not user_input:
                continue
                
            print("\nThinking...")
            answer = get_answer(openai_client, deeplake_client, user_input)
            
            print("-" * 50)
            print(answer)
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
```

**Step 2: Run script to verify it works**

Run: `uv run python augmented_generation.py`
Expected: The script starts, accepts a prompt like "Tell me about space exploration on the Moon and Mars.", prints the debug score, and prints the generated LLM response.

**Step 3: Commit**

```bash
git add augmented_generation.py
git commit -m "feat: add interactive CLI loop for RAG"
```
