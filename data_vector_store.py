import os
import openai
from dotenv import load_dotenv
from deeplake import Client

def chunk_text(text: str, chunk_size: int = 1000) -> list[str]:
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def main():
    load_dotenv()
    
    openai.api_key = os.getenv("OPENAI_API_KEY")
    activeloop_token = os.getenv("ACTIVELOOP_TOKEN")
    
    if not openai.api_key or not activeloop_token:
        print("Missing API keys")
        return
        
    # pyrefly: ignore [unexpected-keyword]
    client = Client(token=activeloop_token, workspace_id="first")
    
    # Drop table if exists for a fresh start
    try:
        client.drop_table("llm_embeddings")
        print("Dropped existing table 'llm_embeddings'.")
    except Exception:
        pass
        
    with open("llm.txt", "r") as f:
        text = f.read()
        
    chunks = chunk_text(text, 1000)
    batch_size = 200
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1} of {(len(chunks)-1)//batch_size + 1}...")
        
        response = openai.embeddings.create(input=batch, model="text-embedding-3-small")
        embeddings = [data.embedding for data in response.data]
        
        client.ingest("llm_embeddings", {"text": batch, "embedding": embeddings})
        
    print("Creating index...")
    client.create_index("llm_embeddings", "embedding")
    client.create_index("llm_embeddings", "text")
    print("Done!")

if __name__ == "__main__":
    main()
