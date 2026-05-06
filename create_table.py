import os
import requests
from dotenv import load_dotenv

load_dotenv()

def create_demo_table():
    token = os.getenv("ACTIVELOOP_TOKEN")
    if not token:
        print("Error: ACTIVELOOP_TOKEN not found in .env")
        return

    workspace = "first"
    table_name = "demo"
    
    # URL for querying/creating tables in the workspace
    url = f"https://api.deeplake.ai/workspaces/{workspace}/tables/query"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Create table with the standard RAG schema
    # Note: embedding size 1536 is for OpenAI text-embedding-3-small or ada-002
    create_query = f"""
    CREATE TABLE {table_name} (
        id TEXT,
        text TEXT,
        metadata JSON,
        embedding FLOAT[1536]
    )
    """
    
    print(f"Creating table '{table_name}' in workspace '{workspace}'...")
    
    try:
        response = requests.post(url, headers=headers, json={"query": create_query})
        
        if response.status_code == 200:
            print(f"Successfully created table '{table_name}'.")
            print("Response:", response.json())
        else:
            print(f"Failed to create table. Status code: {response.status_code}")
            print("Response:", response.text)
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    create_demo_table()
