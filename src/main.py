from src.database.db_client import Neo4jClient
from src.config.config import DB_URI, DB_DATABASE, DB_USER, DB_PASSWORD

from src.services.llm_client import LLMClient
from src.services.ingestion_service import IngestionService, IngestionInput, IngestionSource

if __name__ == '__main__':
    db_client = Neo4jClient(uri=DB_URI, user=DB_USER, password=DB_PASSWORD, database=DB_DATABASE)
    llm_client = LLMClient()
    ingestion_service = IngestionService(db_client=db_client, llm_client=llm_client)


    db_client.init_database()

    with open('./data/gofry.txt', 'r', encoding='utf8') as f:
        waffles = f.read()
        ingestion_service.ingest_recipe(IngestionInput(source=IngestionSource.TEXT, content=waffles, raw_bytes=None))


    db_client.close()
