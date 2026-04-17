# RAG Chapter 01 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Set up local Python environment and adapt the RAG_Overview.ipynb notebook to work locally in VS Code Native Jupyter.

**Architecture:** Python 3.8 with `venv`, `python-dotenv` for API key management, strictly pinned dependencies.

**Tech Stack:** Python 3.8, Jupyter, OpenAI, Scikit-learn, SpaCy, NLTK.

---

### Task 1: Scaffolding Configuration Files

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `README.md`

**Step 1: Create requirements.txt**
Create `requirements.txt` with the following contents:
```text
openai==1.40.3
scikit-learn>=1.3.0
spacy==3.7.4
nltk==3.8.1
numpy>=1.24.0
pandas>=2.0.0
python-dotenv==1.0.0
ipykernel>=6.0.0
```
Run `cat requirements.txt` to verify contents.

**Step 2: Create .gitignore**
Create `.gitignore` to ignore `.venv` and `.env` (or append if exists).
```text
.venv/
.env
__pycache__/
*.pyc
```
Run `cat .gitignore` to verify.

**Step 3: Create .env.example**
Create `.env.example` with:
```text
OPENAI_API_KEY="sk-your-openai-api-key-here"
```
Run `cat .env.example` to verify.

**Step 4: Create README.md**
Create `README.md` with instructions on how to use `.venv`, install requirements, populate `.env` with API key, and select kernel in VS Code.

**Step 5: Commit Configuration Scaffolding**
```bash
git add requirements.txt .gitignore .env.example README.md
git commit -m "chore: add python project config scaffolding"
```

---

### Task 2: Environment Setup

**Files:**
- Modify: None (Command line interactions)

**Step 1: Create Virtual Environment**
Run command: `python3 -m venv .venv`

**Step 2: Verify .venv Creation**
Run command: `ls -la .venv/bin`
Expected: Folder `.venv/bin` should contain Python and Pip executables.

**Step 3: Install Dependencies**
Run command: `.venv/bin/pip install --upgrade pip`
Run command: `.venv/bin/pip install -r requirements.txt`

**Step 4: Download NLP Models**
Run command: `.venv/bin/python -m spacy download en_core_web_sm`
Run command: `.venv/bin/python -c "import nltk; nltk.download('wordnet')"`

**Step 5: Verify Installations**
Run command: `.venv/bin/python -c "import openai, spacy, nltk, sklearn, dotenv; print('All good!')"`
Expected output: `All good!`

---

### Task 3: Jupyter Notebook Copy & Adaptation

**Files:**
- Create: `notebooks/chapter01_rag_overview.ipynb`

**Step 1: Create Notebooks Directory**
Run command: `mkdir -p notebooks`

**Step 2: Copy Original Notebook**
Run command: `cp /home/mahmoud/courses/AI-Books/rag-drivengenerativeai/RAG-Driven-Generative-AI-main/Chapter01/RAG_Overview.ipynb notebooks/chapter01_rag_overview.ipynb`

**Step 3: Verify Notebook Copied**
Run command: `ls -la notebooks/`

**Step 4: Adapt Notebook (Replace Colab Auth)**
Using Python script over Jupyter notebook JSON (or manual file replacement tools), replace the Google Colab Drive mounting code with `dotenv` API key loader.

Target cells to replace:
```python
from google.colab import drive
drive.mount('/content/drive')
```
and
```python
f = open("drive/MyDrive/files/api_key.txt", "r")
API_KEY=f.readline().strip()
f.close()

#The OpenAI Key
import os
import openai
os.environ['OPENAI_API_KEY'] =API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")
```

Replace with:
```python
import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
```

**Step 5: Commit Notebook**
```bash
git add notebooks/chapter01_rag_overview.ipynb
git commit -m "feat: add and adapt Chapter 01 RAG notebook"
```
