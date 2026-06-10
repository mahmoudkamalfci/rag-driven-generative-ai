"""Pipeline 2: Create DeepLake vector store using LlamaIndex integration."""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

# pyrefly: ignore [missing-import]
from config import DEEPLAKE_DATASET_PATH, DATA_DIR, deeplake_client  # noqa: E402

from llama_index.core import (  # noqa: E402
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.vector_stores.deeplake import DeepLakeVectorStore  # noqa: E402
from llama_index.embeddings.openai import OpenAIEmbedding  # noqa: E402
from llama_index.llms.openai import OpenAI  # noqa: E402

# Configure LlamaIndex
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)


def main():
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print("No data files found. Run pipeline_1_collect_data.py first.")
        return

    print(f"Loading documents from {DATA_DIR}...")
    documents = SimpleDirectoryReader(
        input_dir=DATA_DIR,
        required_exts=[".txt"]
    ).load_data()
    print(f"Loaded {len(documents)} documents.")

    print(f"Creating DeepLake vector store at {DEEPLAKE_DATASET_PATH}...")
    
    # Drop and recreate/register the table via managed API to make it visible in the UI
    try:
        print("Dropping existing table in cloud control plane (if any)...")
        deeplake_client.drop_table("visdrone_deeplake_store")
    except Exception:
        pass
    
    print("Registering new table with DeepLake cloud control plane...")
    deeplake_client._create_table_via_api(
        "visdrone_deeplake_store",
        DEEPLAKE_DATASET_PATH,
        {"text": "TEXT", "metadata": "JSONB", "embedding": "float4[]", "id": "TEXT"}
    )

    vector_store = DeepLakeVectorStore(
        dataset_path=DEEPLAKE_DATASET_PATH,
        overwrite=True,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Building VectorStoreIndex (embedding documents)...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    print(f"\nVector store created at: {DEEPLAKE_DATASET_PATH}")
    print(f"Documents indexed: {len(documents)}")


if __name__ == "__main__":
    main()
