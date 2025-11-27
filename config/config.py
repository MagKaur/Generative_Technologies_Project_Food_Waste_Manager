import os
import logging
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("DB_USER", "neo4j")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_URI = os.getenv("DB_URI", "neo4j://localhost:7687")
DB_DATABASE = os.getenv("DB_DATABASE", "neo4j")

RAW_DATASET_PATH = os.getenv("RAW_DATASET_PATH", "")
DATASET_PATH = os.getenv("DATASET_PATH", "")

AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-nano")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-12-01-preview")



# logging
#logging.basicConfig(level=logging.DEBUG)
