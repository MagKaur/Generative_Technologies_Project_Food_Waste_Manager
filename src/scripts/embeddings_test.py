from clients.llm_client import LLMClient
from src.database.database_service import DatabaseService
from src.config.config import DB_URI, DB_USER, DB_PASSWORD, DB_DATABASE

def main():
    db = DatabaseService(uri=DB_URI, user=DB_USER, password=DB_PASSWORD, database=DB_DATABASE)
    llm = LLMClient()
    vec = llm.embed_text("test")
    print(len(vec))

if __name__ == "__main__":
    main()