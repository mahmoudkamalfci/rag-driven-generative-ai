"""Pipeline 3: RAG query engines — VectorStoreIndex + TreeIndex."""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

# pyrefly: ignore [missing-import]
from config import DEEPLAKE_DATASET_PATH, DATA_DIR, TREE_INDEX_DIR  # noqa: E402

from llama_index.core import (  # noqa: E402
    VectorStoreIndex,
    TreeIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.vector_stores.deeplake import DeepLakeVectorStore  # noqa: E402
from llama_index.embeddings.openai import OpenAIEmbedding  # noqa: E402
from llama_index.llms.openai import OpenAI  # noqa: E402

# Configure LlamaIndex
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)


def build_vector_query_engine():
    """Load DeepLake vector store and create a query engine."""
    print("Loading vector store from DeepLake...")
    vector_store = DeepLakeVectorStore(
        dataset_path=DEEPLAKE_DATASET_PATH,
        read_only=True
    )
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index.as_query_engine(
        similarity_top_k=5,
        response_mode="compact",
    )


def build_tree_query_engine():
    """Build or load a tree index and create a query engine."""
    if os.path.exists(TREE_INDEX_DIR):
        print("Loading existing tree index...")
        storage_context = StorageContext.from_defaults(persist_dir=TREE_INDEX_DIR)
        index = load_index_from_storage(storage_context)
    else:
        print("Building tree index from documents (first run)...")
        documents = SimpleDirectoryReader(
            input_dir=DATA_DIR,
            required_exts=[".txt"],
        ).load_data()

        index = TreeIndex.from_documents(documents, show_progress=True)

        os.makedirs(TREE_INDEX_DIR, exist_ok=True)
        index.storage_context.persist(persist_dir=TREE_INDEX_DIR)
        print(f"Tree index saved to {TREE_INDEX_DIR}")

    return index.as_query_engine(response_mode="tree_summarize")


def main():
    print("=" * 60)
    print("  VisDrone RAG Query Engine")
    print("=" * 60)

    vector_engine = build_vector_query_engine()
    tree_engine = build_tree_query_engine()

    print("\nReady! Options:")
    print("  [1] Vector Store Index (similarity search)")
    print("  [2] Tree Index (hierarchical summarization)")
    print("  [q] Quit\n")

    while True:
        try:
            choice = input("Engine [1/2/q]: ").strip()
            if choice.lower() == 'q':
                print("Goodbye!")
                break

            if choice not in ('1', '2'):
                print("Invalid — enter 1, 2, or q.")
                continue

            query = input("Question: ").strip()
            if not query:
                continue

            print("\nThinking...")
            if choice == '1':
                response = vector_engine.query(query)
                label = "Vector Store Index"
            else:
                response = tree_engine.query(query)
                label = "Tree Index"

            print(f"\n[{label}]")
            print("-" * 50)
            print(response)
            print("-" * 50)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
