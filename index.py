# index.py
import numpy as np
from openai import OpenAI
from typing import List, Tuple

from dotenv import load_dotenv

load_dotenv()


def load_and_chunk(
    file_path: str, chunk_size: int = 500, overlap: int = 50
) -> List[Tuple[str, int]]:
    """Read a text file and split into fixed-size chunks with overlap.

    Returns a list of (chunk_text, chunk_index) tuples.
    Raises ValueError if the file is empty.
    """
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


def embed_batch(client: OpenAI, texts: List[str]) -> List[List[float]]:
    """Call OpenAI embeddings API for a batch of texts.

    Returns a list of 1536-dim float32 vectors.
    Raises openai.APIError on failure.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


def store_to_deeplake(
    ds, texts: List[str], metadatas: List[dict], embeddings: List[List[float]]
) -> None:
    """Append one batch of chunks to all three DeepLake tensors."""
    with ds:
        for text, meta, emb in zip(texts, metadatas, embeddings):
            ds.text.append(text)
            ds.metadata.append(meta)
            ds.embedding.append(np.array(emb, dtype="float32"))