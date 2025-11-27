import ast
import pandas as pd
import kagglehub
from config.config import DATASET_PATH, DB_URI, DB_USER, DB_PASSWORD, DB_DATABASE
from llm_client import LLMClient
from src.database_scripts.db_client import Neo4jClient


def download_dataset() -> str:
    return kagglehub.dataset_download("wilmerarltstrmberg/recipe-dataset-over-2m")

def extract_and_save_dataset(original_path: str, save_path: str, numer_of_rows: int):
    pd.read_csv(original_path, nrows=numer_of_rows).to_csv(save_path, index=False)

def str_to_list_dict(cell):
    try:
        val = ast.literal_eval(cell)
        if isinstance(val, list):
            return val
        else:
            return []
    except Exception:
        return []


def init_all():
    """You need to use it once. This function downloads dataset from kaggle, parsing with llm, creating necessary
    indexes in db and inserting data"""
    # Download and extract data form original dataset
    dataset_path = download_dataset()
    extract_and_save_dataset(dataset_path, DATASET_PATH, numer_of_rows=200)

    # Load extracted data do pandas.DataFrame
    df = pd.read_csv(DATASET_PATH)

    # Extract unstructured data to structured from
    llm_client = LLMClient()
    df['ingredients'] = df['ingredients'].apply(llm_client.parse_ingredients)

    # Insert to Neo4j database
    client = Neo4jClient(DB_URI, DB_USER, DB_PASSWORD, DB_DATABASE)
    client.init_database()
    client.load_data(df)




