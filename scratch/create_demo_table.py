import os
from dotenv import load_dotenv
from deeplake import Client

load_dotenv()

token = os.getenv("ACTIVELOOP_TOKEN")
workspace = "first" # As per user request "workspace: first"

print(f"Connecting to workspace '{workspace}'...")
client = Client(token=token, workspace_id=workspace)

# Create a demo table with some sample data
table_name = "demo1"
data = {
    "text": ["Initial data for demo table"],
    "metadata": [{"info": "table created via deeplake skill"}]
}

print(f"Creating table '{table_name}'...")
try:
    # First, drop if it exists to ensure it's "new" as requested
    client.drop_table(table_name, if_exists=True)
    
    result = client.ingest(table_name, data)
    print(f"Successfully created table: {result}")
except Exception as e:
    print(f"Error: {e}")
