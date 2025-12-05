import ast
import pandas as pd
import kagglehub
from src.config.config import DATASET_PATH, DB_URI, DB_USER, DB_PASSWORD, DB_DATABASE
from src.database_scripts.llm_client import LLMClient
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
    # dataset_path = download_dataset()
    # extract_and_save_dataset(dataset_path, DATASET_PATH, numer_of_rows=200)

    # Load extracted data do pandas.DataFrame
    df = pd.read_csv(DATASET_PATH)

    # Extract unstructured data to structured from
    llm_client = LLMClient()
    df['ingredients'] = df['ingredients'].apply(llm_client.parse_ingredients)

    # concatenated text for embeddings: title + joined ingredients
    def create_recipe_text(row):
        ingredients_str = ", ".join(
            [f"{ing.get('amount', '')} {ing.get('unit', '')} {ing['name']}" for ing in row['ingredients']])
        return f"{row['title']}: {ingredients_str}"

    df['recipe_text'] = df.apply(create_recipe_text, axis=1)

    # Create embeddings
    texts = df['recipe_text'].tolist()
    embeddings = llm_client.generate_embeddings_batch(texts)
    df['recipe_embedding'] = embeddings

    # Insert to Neo4j database
    client = Neo4jClient(DB_URI, DB_USER, DB_PASSWORD, DB_DATABASE)
    client.init_database()
    client.load_data(df)


def chatbot_suggest_recipes(llm_client, db_client, user_query: str):
    """
    Przykład: user_query = "mam jajka, mleko i mąkę co mogę z tego zrobić"
    """
    # 1. Parsuj składniki z query (użyj parse_ingredients lub prosty split)
    user_ingredients = ["brown sugar", "butter", "eggs", "vanilla"]  # W realu: LLM parsuje query

    # 2. Generuj embedding dla query
    query_embedding = llm_client.generate_query_embedding(user_query)

    # 3. Hybrydowe wyszukiwanie
    similar = db_client.query_similar_recipes_hybrid(query_embedding, user_ingredients, min_matches=2, top_k=3)

    # 4. Wyświetl (lub wyślij do LLM na generację odpowiedzi)
    for rec in similar:
        print(f"Przepis: {rec['title']} (podobieństwo: {rec['score']:.2f}, pasujące składniki: {rec['matches']})")

    return similar



