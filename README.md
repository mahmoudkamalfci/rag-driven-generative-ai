# RAG-Driven Generative AI - Chapter 01

This project runs the Jupyter Notebook from Chapter 01 of the book locally using VS Code Native Jupyter.

## Setup Instructions

1. **Activate the Virtual Environment**
   This project uses Python 3.8.10 with `venv`.
   ```bash
   source .venv/bin/activate
   ```

2. **Add Your OpenAI API Key**
   Copy the example template:
   ```bash
   cp .env.example .env
   ```
   Open `.env` and add your real `OPENAI_API_KEY`.

3. **Run in VS Code**
   - Open `notebooks/chapter01_rag_overview.ipynb`
   - In the top right of VS Code, click "Select Kernel" -> "Python Environments" -> choose `.venv`
   - Run the cells!
