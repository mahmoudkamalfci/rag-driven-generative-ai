import os
import openai
from dotenv import load_dotenv
from deeplake import Client

# Minimum cosine similarity score to consider a retrieved chunk as relevant.
# Scores range from -1 (opposite) to 1 (identical). Adjust this value as needed.
RELEVANCE_THRESHOLD = 0.3

def augment_prompt(user_prompt: str, context: str) -> str:
    return f"{user_prompt}\n\nContext:\n{context}"

# pyrefly: ignore [unexpected-keyword]
def setup_clients():
    load_dotenv()
    
    openai_key = os.getenv("OPENAI_API_KEY")
    activeloop_token = os.getenv("ACTIVELOOP_TOKEN")
    
    if not openai_key or not activeloop_token:
        return None, None
        
    openai.api_key = openai_key
    # pyrefly: ignore [unexpected-keyword]
    client = Client(token=activeloop_token, workspace_id="first")
    return openai, client

def get_answer(openai_client, deeplake_client, query: str) -> str:
    # 1. Embed query
    response = openai_client.embeddings.create(input=[query], model="text-embedding-3-small")
    query_embedding = response.data[0].embedding
    
    # 2. Search DeepLake using SQL query
    # Using ORDER BY similarity DESC to get the highest cosine similarity
    # Note: <#> represents cosine similarity in pg_deeplake
    results = deeplake_client.query("""
        SELECT text, embedding <#> $1 AS similarity
        FROM llm_embeddings
        ORDER BY similarity DESC
        LIMIT 1
    """, (query_embedding,))
    
    # Check if we have results
    if not results:
        return "No relevant context found."
        
    top_text = results[0]["text"]
    top_score = results[0]["similarity"]
    print(f"\n[Debug] Context Match Score: {top_score:.4f}\n")

    # 3. Relevance gate — skip LLM call if context is not relevant enough
    if top_score < RELEVANCE_THRESHOLD:
        return "No relevant information available for your query."

    # 4. Augment Prompt
    augmented = augment_prompt(query, top_text)
    
    # 4. Ask LLM
    chat_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a space exploration expert. Use the provided context to answer the user's question in detail."},
            {"role": "user", "content": augmented}
        ],
        temperature=0.1
    )
    return chat_response.choices[0].message.content

def main():
    openai_client, deeplake_client = setup_clients()
    if not openai_client or not deeplake_client:
        print("Missing environment variables. Make sure OPENAI_API_KEY and ACTIVELOOP_TOKEN are set in .env")
        return
        
    print("Welcome to the Space Exploration RAG CLI!")
    print("Type 'quit' or 'exit' to stop.")
    
    while True:
        try:
            user_input = input("\nEnter your question: ").strip()
            if user_input.lower() in ['quit', 'exit']:
                break
                
            if not user_input:
                continue
                
            print("\nThinking...")
            answer = get_answer(openai_client, deeplake_client, user_input)
            
            print("-" * 50)
            print(answer)
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
