# index.py
import os

from dotenv import load_dotenv

load_dotenv()

# 1. Validate credentials
openai_key = os.getenv("OPENAI_API_KEY")
activeloop_token = os.getenv("ACTIVELOOP_TOKEN")

