"""Shared configuration — loads credentials from parent .env file."""
import os
from dotenv import load_dotenv

# Load .env from project root (parent of this folder)
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ACTIVELOOP_TOKEN = os.getenv("ACTIVELOOP_TOKEN")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")
if not ACTIVELOOP_TOKEN:
    raise ValueError("ACTIVELOOP_TOKEN not found in .env")

# Set for LlamaIndex auto-detection
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["ACTIVELOOP_TOKEN"] = ACTIVELOOP_TOKEN

# Initialize DeepLake Client to fetch cloud path and apply temporary credentials
from deeplake import Client
deeplake_client = Client(token=ACTIVELOOP_TOKEN, workspace_id="first")  # type: ignore
deeplake_client._apply_storage_creds()

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
DEEPLAKE_DATASET_PATH = deeplake_client._get_dataset_path("visdrone_deeplake_store")
TREE_INDEX_DIR = os.path.join(BASE_DIR, "indexes", "tree_index")
