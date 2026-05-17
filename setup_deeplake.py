import os
import requests
from dotenv import load_dotenv
from deeplake import Client
from deeplake.managed import TableError, WorkspaceError

load_dotenv()

activeloop_token = os.getenv("ACTIVELOOP_TOKEN")
if not activeloop_token:
    print("Error: ACTIVELOOP_TOKEN not found in environment.")
    exit(1)

api_url = "https://api.deeplake.ai"

print("Creating workspace 'first'...")
try:
    resp = requests.post(
        f"{api_url}/workspaces",
        headers={"Authorization": f"Bearer {activeloop_token}"},
        json={"id": "first", "name": "First Workspace"},
    )
    if resp.status_code == 409:
        print("Workspace 'first' already exists.")
    else:
        resp.raise_for_status()
        print("Workspace 'first' created successfully.")
except requests.exceptions.RequestException as e:
    print(f"Failed to create workspace: {e}")
    if e.response is not None:
        print(e.response.text)
    # Proceed anyway in case it already exists

print("\nCreating table 'my_first_table'...")
client = Client(token=activeloop_token, workspace_id="first")

try:
    client.ingest("my_first_table", {"dummy_col": ["Hello World"]}, schema={"dummy_col": "TEXT"})
    print("Table 'my_first_table' created successfully!")
except Exception as e:
    print(f"Error creating table: {e}")
