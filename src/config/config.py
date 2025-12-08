import os
import logging
from dotenv import load_dotenv

load_dotenv(dotenv_path='C:\\Users\\dawid\\PycharmProjects\\Generative_Technologies_Project_Food_Waste_Manager\\src\\config\\.env')

DB_USER = os.getenv("DB_USER", "neo4j")
DB_PASSWORD = os.getenv("DB_PASSWORD", "neo4j")
DB_URI = os.getenv("DB_URI","")
DB_DATABASE = os.getenv("DB_DATABASE", "neo4j")

RAW_DATASET_PATH = os.getenv("RAW_DATASET_PATH", "")
DATASET_PATH = os.getenv("DATASET_PATH", "")

AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-nano")
AZURE_OPENAI_EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-12-01-preview")



LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
LOG_FILE = os.getenv("LOG_FILE", "app.log")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# logging.basicConfig(
#     level=getattr(logging, LOG_LEVEL),
#     filename=LOG_FILE,
#     filemode='a',
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )
