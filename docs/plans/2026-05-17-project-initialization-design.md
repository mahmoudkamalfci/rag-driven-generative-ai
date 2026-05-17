# Project Initialization Design

## Architecture / Setup
We are migrating/setting up a new Python project in the existing `rag-driven-generative-ai` repository. The goal is to create a Python 3.15 virtual environment and maintain a precise record of all dependencies and their exact versions.

We chose to use `uv` (version 0.11.8) as the modern, fast package manager for this task.

## Implementation Steps
1. **Virtual Environment Setup:**
   - Execute `uv venv --python 3.15 .venv` in the root directory.
   - This explicitly creates a lightweight, isolated Python environment targeting Python 3.15.

2. **Dependencies Configuration:**
   - Execute `uv init` in the root directory.
   - This will generate a `pyproject.toml` file, establishing the standard for top-level project dependencies in modern Python projects.

3. **Lockfile / Version Tracking (The Dependencies File):**
   - Going forward, any package added via `uv add <package>` will automatically generate or update the `uv.lock` file.
   - The `uv.lock` file serves as the strict dependency tracker, meeting the requirement to check all installed packages (direct and sub-dependencies) and their exact, locked versions.
