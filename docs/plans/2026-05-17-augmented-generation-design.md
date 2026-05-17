# Augmented Generation Design

## Overview
This document outlines the design for converting `3_Augmented_Generation.ipynb` into a standalone, interactive Python script named `augmented_generation.py`. The script implements a stateless interactive CLI for Retrieval-Augmented Generation (RAG).

## Architecture & Flow

### 1. Environment & Connections
- **Environment Variables**: Load `OPENAI_API_KEY` and `ACTIVELOOP_TOKEN` via `python-dotenv`.
- **DeepLake Store**: Instantiate a DeepLake `VectorStore` pointing to `workspace_id="first"` and the `llm_embeddings` dataset previously created.
- **OpenAI Client**: Initialize the official OpenAI Python SDK client.

### 2. Retrieval & Augmentation
- **Interactive Loop**: Run a `while True:` loop prompting the user for questions via `input()`. The user can exit by typing 'quit' or 'exit'.
- **Embedding Generation**: Convert the user's input into embeddings using `openai.embeddings.create` with the `text-embedding-3-small` model.
- **Vector Search**: Use the generated embedding to query the DeepLake vector store and fetch the top scoring text result.
- **Prompt Augmentation**: Combine the original user prompt with the retrieved context into a single string:
  `"{user_prompt}\n\nContext:\n{retrieved_text}"`

### 3. LLM Generation & Output
- **Model**: Use `gpt-4o-mini` for fast and cost-effective generation.
- **System Prompt**: Set a system behavior message, e.g., `"You are a space exploration expert. Use the provided context to answer the user's question in detail."`
- **Output Presentation**: Print the generated response clearly to the console. Optionally, print the retrieved context's score to provide transparency on retrieval confidence.
