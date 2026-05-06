
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
    """Append one batch of chunks to all three DeepLake columns."""
    ds.append({
        "text": texts,
        "metadata": metadatas,
        "embedding": np.array(embeddings, dtype="float32")
    })


deeplake.client.endpoint = "https://api.deeplake.ai"
DATASET_PATH = "al://first-space/my_table"
SOURCE_FILE = "notebooks/chapter-2/llm.txt"
CHUNK_SIZE = 500
OVERLAP = 50
BATCH_SIZE = 200


def main() -> None:
    # 1. Validate credentials
    openai_key = os.getenv("OPENAI_API_KEY")
    activeloop_token = os.getenv("ACTIVELOOP_TOKEN")
    if not openai_key:
        raise EnvironmentError("OPENAI_API_KEY not found in environment / .env")
    if not activeloop_token:
        raise EnvironmentError("ACTIVELOOP_TOKEN not found in environment / .env")

    # 2. Init clients
    client = OpenAI(api_key=openai_key)

    # 3. Load & chunk
    print(f"Loading and chunking '{SOURCE_FILE}' ...")
    chunks = load_and_chunk(SOURCE_FILE, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    print(f"  -> {len(chunks)} chunks created")

    # 4. Connect to DeepLake (overwrite existing data)
    # 1. Delete existing table via API
    requests.delete(
        f"https://api.deeplake.ai/workspaces/first-space/tables/my_table",
        headers={"Authorization": f"Bearer {activeloop_token}"}
    )

    # 2. Create table with explicit schema via SQL query API
    # This ensures the dashboard/SQL engine sees the columns immediately.
    print("Creating table with explicit schema via SQL API...")
    create_query = (
        "CREATE TABLE my_table ("
        "text TEXT, "
        "metadata JSON, "
        "embedding FLOAT[1536]"
        ")"
    )
    resp = requests.post(
        "https://api.deeplake.ai/workspaces/first-space/tables/query",
        headers={"Authorization": f"Bearer {activeloop_token}"},
        json={"query": create_query}
    )
    resp.raise_for_status()

    # 3. Open the dataset for ingestion
    print(f"Opening DeepLake dataset: {DATASET_PATH}")
    ds = deeplake.open(DATASET_PATH, token=activeloop_token)
    # Columns are already created by the SQL query above.

    # 5. Batch embed & store
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_num, batch_start in enumerate(range(0, len(chunks), BATCH_SIZE), start=1):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        texts = [c[0] for c in batch]
        metadatas = [{"source": SOURCE_FILE, "chunk_index": c[1]} for c in batch]

        print(
            f"  Embedding batch {batch_num}/{total_batches} " f"({len(texts)} chunks) ...",
            end=" ",
            flush=True,
        )

        try:
            embeddings = embed_batch(client, texts)
        except Exception as e:
            print(f"\nBatch {batch_num} failed: {e}")
            raise

        store_to_deeplake(ds, texts, metadatas, embeddings)
        chunks_stored = batch_start + len(texts)
        print(f"done  ({chunks_stored}/{len(chunks)} stored)")

    print(f"\nDone! {len(chunks)} chunks embedded and stored to {DATASET_PATH}")


if __name__ == "__main__":
    main()