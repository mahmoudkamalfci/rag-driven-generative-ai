# Data Collection Preparation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert the data collection Jupyter Notebook into a reusable Python script that fetches, cleans, and saves Wikipedia articles to `llm.txt`.

**Architecture:** A modular Python script with isolated functions for cleaning text and fetching articles, orchestrated by a main block.

**Tech Stack:** Python, `requests`, `beautifulsoup4`, `pytest` (for TDD).

---

### Task 1: Environment Setup

**Files:**
- Modify: `pyproject.toml` or directly use `.venv`

**Step 1: Install dependencies**
Run: `uv pip install beautifulsoup4 requests pytest`
Expected: Dependencies successfully installed in `.venv`.

**Step 2: Commit**
```bash
git commit -m "chore: install dependencies for data collection" --allow-empty
```

### Task 2: Core Utility - `clean_text`

**Files:**
- Create: `data_collection_preparation.py`
- Create: `test_data_collection.py`

**Step 1: Write the failing test**

Modify `test_data_collection.py`:
```python
from data_collection_preparation import clean_text

def test_clean_text():
    raw = "Space exploration is the use of astronomy [1] and space technology [2]."
    expected = "Space exploration is the use of astronomy  and space technology ."
    assert clean_text(raw) == expected
```

**Step 2: Run test to verify it fails**
Run: `uv run pytest test_data_collection.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`.

**Step 3: Write minimal implementation**

Modify `data_collection_preparation.py`:
```python
import re

def clean_text(content: str) -> str:
    """Remove references that usually appear as [1], [2], etc."""
    return re.sub(r'\[\d+\]', '', content)
```

**Step 4: Run test to verify it passes**
Run: `uv run pytest test_data_collection.py -v`
Expected: PASS

**Step 5: Commit**
```bash
git add data_collection_preparation.py test_data_collection.py
git commit -m "feat: add clean_text utility function"
```

### Task 3: Core Utility - `fetch_and_clean_article`

**Files:**
- Modify: `data_collection_preparation.py`

*(Note: We will skip mocking the network request in tests for simplicity and directly implement the function, verifying by running the script later.)*

**Step 1: Write implementation**

Modify `data_collection_preparation.py` (Append this):
```python
import requests
from bs4 import BeautifulSoup

def fetch_and_clean_article(url: str) -> str:
    """Fetch Wikipedia URL and extract cleaned article text."""
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the main content
    content = soup.find('div', {'class': 'mw-parser-output'})
    if not content:
        return ""
        
    # Remove the bibliography/reference sections
    for section_title in ['References', 'Bibliography', 'External links', 'See also']:
        section = content.find('span', id=section_title)
        if section:
            for sib in section.parent.find_next_siblings():
                sib.decompose()
            section.parent.decompose()
            
    text = content.get_text(separator=' ', strip=True)
    return clean_text(text)
```

**Step 2: Commit**
```bash
git add data_collection_preparation.py
git commit -m "feat: add fetch_and_clean_article function"
```

### Task 4: Main Orchestration Block

**Files:**
- Modify: `data_collection_preparation.py`

**Step 1: Write implementation**

Modify `data_collection_preparation.py` (Append this):
```python
URLS = [
    "https://en.wikipedia.org/wiki/Space_exploration",
    "https://en.wikipedia.org/wiki/Apollo_program",
    # We will include all URLs from the notebook here, truncated for brevity in this plan snippet
]

def main():
    print(f"Starting data collection for {len(URLS)} articles...")
    with open('llm.txt', 'w', encoding='utf-8') as file:
        for url in URLS:
            print(f"Fetching: {url}")
            clean_article_text = fetch_and_clean_article(url)
            file.write(clean_article_text + '\\n')
    print("Content written to llm.txt successfully.")

if __name__ == "__main__":
    main()
```

*(Note: The actual implementation will need to copy the full `urls` list from the notebook).*

**Step 2: Run the script to verify**
Run: `uv run python data_collection_preparation.py`
Expected: Output showing articles being fetched and `llm.txt` created successfully.

**Step 3: Commit**
```bash
git add data_collection_preparation.py
git commit -m "feat: add main orchestrator and URLs to data collection script"
```
