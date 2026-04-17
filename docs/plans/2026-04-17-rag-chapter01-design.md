# RAG-Driven Generative AI — Chapter 01 Local Setup Design

**Date:** 2026-04-17
**Author:** Mahmoud
**Source Book:** *RAG-Driven Generative AI* by Denis Rothman
**Notebook:** `Chapter01/RAG_Overview.ipynb`

---

## 1. Goal

Set up the book's Chapter 01 Jupyter notebook to run locally in **VS Code Native Jupyter**, replacing the original Google Colab environment. The notebook covers Naïve RAG, Advanced RAG, and Modular RAG using OpenAI GPT-4o.

---

## 2. Architecture

### Project Structure

```
rag-driven-generative-ai/
├── .venv/                              ← Python 3.8 virtual environment
├── .env                                ← OpenAI API key (gitignored)
├── .env.example                        ← Template for required env vars
├── .gitignore                          ← Protects secrets & venv
├── requirements.txt                    ← Pinned Python dependencies
├── notebooks/
│   └── chapter01_rag_overview.ipynb   ← Adapted notebook (local-ready)
└── README.md                           ← Setup & run instructions
```

### Key Design Decisions

- **`venv`** — Python's built-in virtual environment, no extra tooling, auto-detected by VS Code
- **`python-dotenv`** — replaces Google Colab's Drive mount for secure API key loading
- **`notebooks/` folder** — keeps the workspace organized for future chapters (Ch02–Ch10)
- **Pinned `requirements.txt`** — ensures reproducible installs matching the book's versions

---

## 3. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `openai` | `==1.40.3` | OpenAI GPT-4o API (exact version per book) |
| `scikit-learn` | `>=1.3.0` | TF-IDF vectorizer + cosine similarity |
| `spacy` | `==3.7.4` | Lemmatization, stop word removal |
| `nltk` | `==3.8.1` | WordNet synonyms for enhanced similarity |
| `numpy` | `>=1.24.0` | Vector math |
| `pandas` | `>=2.0.0` | TF-IDF feature display (DataFrame) |
| `python-dotenv` | `==1.0.0` | Load `.env` for API key |
| `ipykernel` | `>=6.0.0` | VS Code Jupyter kernel registration |

### Post-install Downloads
```bash
python -m spacy download en_core_web_sm      # spaCy English model
python -c "import nltk; nltk.download('wordnet')"  # NLTK WordNet
```

---

## 4. Notebook Adaptation

The only change needed to the original notebook is replacing the 3 Google Colab-specific cells:

**Remove (Colab only):**
```python
from google.colab import drive
drive.mount('/content/drive')
f = open("drive/MyDrive/files/api_key.txt", "r")
API_KEY = f.readline().strip()
f.close()
import os
import openai
os.environ['OPENAI_API_KEY'] = API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")
```

**Replace with (local):**
```python
import os
import openai
from dotenv import load_dotenv

load_dotenv()  # loads .env from project root
openai.api_key = os.getenv("OPENAI_API_KEY")
```

All other notebook content stays **exactly as the book intended**.

---

## 5. Setup Flow (Execution Order)

### Phase 1 — Environment Setup
1. Create `.venv` inside the project folder
2. Activate virtual environment
3. Upgrade pip
4. Install all packages from `requirements.txt`
5. Download spaCy `en_core_web_sm` model
6. Download NLTK `wordnet` data
7. Register `.venv` as a Jupyter kernel in VS Code

### Phase 2 — Project Scaffolding
8. Create `requirements.txt`
9. Create `.env.example`
10. Create `.env` (user adds real API key)
11. Create `.gitignore`
12. Create `notebooks/` folder
13. Create adapted `chapter01_rag_overview.ipynb`
14. Create `README.md`

### Phase 3 — Verification
15. Validate all imports work
16. Confirm VS Code detects the kernel
17. Run first notebook cell — verify OpenAI connection

---

## 6. Error Handling

| Issue | Solution |
|-------|----------|
| OpenAI API key missing | `python-dotenv` + OS env check raises clear error before any API call |
| spaCy model not found | Import error caught at cell level with clear download instruction |
| Package version conflict | Pinned `requirements.txt` prevents drift |
| Wrong kernel selected | README reminder + kernel labeled `Python 3.8.10 (.venv)` |
| Python 3.8 EOL | Acceptable for practice; note in README to upgrade for production |

---

## 7. What This Notebook Teaches (Chapter 01 Summary)

| Section | Technique | Retrieval Method |
|---------|-----------|-----------------|
| Part 1 | Foundation | No retrieval (bare LLM) |
| Part 2.1 | Naïve RAG | Keyword matching |
| Part 2.2 | Advanced RAG | TF-IDF cosine similarity vector search |
| Part 2.3 | Advanced RAG | Index-based TF-IDF search |
| Part 2.4 | Modular RAG | Pluggable `RetrievalComponent` class (keyword/vector/indexed) |
