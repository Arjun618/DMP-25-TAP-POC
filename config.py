"""
Configuration file for The Apprentice Project (TAP) AI Assistant
"""
import os
from dotenv import load_dotenv
import requests
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Utility to fetch current date from the internet
# Falls back to local date if request fails

def get_current_date_online():
    try:
        response = requests.get("https://worldtimeapi.org/api/timezone/Etc/UTC", timeout=5)
        if response.status_code == 200:
            data = response.json()
            utc_date = data["utc_datetime"][:10]  # Format: YYYY-MM-DD
            return utc_date
    except Exception:
        pass
    # Fallback to local date
    return datetime.utcnow().strftime("%Y-%m-%d")

# Global constants
DATA_DIR = "data"
INDEX_PATH = "faiss_index"
REFERENCE_DATE = get_current_date_online()  # Current date from internet or local
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LLM Configuration
USE_LOCAL_MODEL = False  # Set to False to use Hugging Face API
LLM_MODEL = "google/flan-t5-base"  # Use Flan-T5-Small via Hugging Face (public, supported model)
LOCAL_MODEL_PATH = os.path.join("models", "llama-2-7b-chat.Q4_0.gguf")  # Path to local model file
LOCAL_MODEL_TYPE = "llama"  # Model type for CTransformers: llama, gpt2, gpt_neox, etc.
MODEL_MAX_TOKENS = 512  # Maximum tokens for LLM response

# Get Hugging Face API token from environment (only needed if USE_LOCAL_MODEL is False)
HUGGINGFACE_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")