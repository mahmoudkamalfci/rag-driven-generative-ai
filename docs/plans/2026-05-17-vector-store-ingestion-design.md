# Design: Batched Vector Store Ingestion

## Overview
This document outlines the design for `data_vector_store.py`, a script responsible for reading local text data, generating embeddings using OpenAI, and storing them in DeepLake. 

## Approach
We will use a **Batched Processing** approach. Instead of attempting to embed and ingest the entire file in a single API call (which can hit OpenAI payload limits), the text will be processed in smaller batches.

## Details

1. **Dependencies & Configuration:**
   - OpenAI package will be updated/installed via `uv add openai`.
   - Environment variables (`OPENAI_API_KEY`, `ACTIVELOOP_TOKEN`) loaded from `.env`.
   - Clients for OpenAI and DeepLake will be initialized, targeting the `first` workspace in DeepLake.

2. **Data Processing:**
   - Source data: `llm.txt`.
   - Chunking Strategy: Split into chunks of 1,000 characters.

3. **Batched Embedding & Ingestion:**
   - Target Table: `llm_embeddings`.
   - If the table already exists, it will be dropped at the start of the script to ensure a fresh run.
   - Batch size: 200 chunks per batch.
   - For each batch:
     - Generate embeddings using OpenAI's `text-embedding-3-small` model.
     - Ingest the batch of text and embeddings into DeepLake using `client.ingest()`.

4. **Indexing:**
   - After ingestion is complete, a vector index will be created on the `embedding` column by calling `client.create_index("llm_embeddings", "embedding")`. This enables fast vector similarity searches.
