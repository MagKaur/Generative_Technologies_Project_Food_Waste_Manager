import csv
import logging
from pathlib import Path

from src.services.ingestion_service import IngestionService, IngestionInput, IngestionSource
from src.database.db_client import Neo4jClient
from src.services.llm_client import LLMClient
from src.config.config import DB_URI, DB_USER, DB_PASSWORD, DB_DATABASE, RAW_DATASET_PATH

logger = logging.getLogger(__name__)


def row_to_text(row: dict, sep: str = "\n") -> str:
    """Joins csv columns into single string."""
    parts = []
    for col, val in row.items():
        if val is None:
            continue
        text = str(val).strip()
        if not text:
            continue
        parts.append(f"{text}")
    return sep.join(parts)


def init_database_from_csv(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"File {csv_path} not found")

    db = Neo4jClient(uri=DB_URI, user=DB_USER, password=DB_PASSWORD, database=DB_DATABASE)
    db.init_database()

    llm = LLMClient()
    ingestion = IngestionService(db_client=db, llm_client=llm)

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            try:
                full_text = row_to_text(row)
                input_ = IngestionInput(
                    source=IngestionSource.TEXT,
                    content=full_text,
                )

                ingestion.ingest_recipe(input_)

                if i % 20 == 0:
                    logger.info("Loaded %d recipes...", i)
            except Exception as e:
                logger.exception("Error in line %d: %s", i, e)

    logger.info("Init DB from CSV successfully ended.")


if __name__ == "__main__":
    init_database_from_csv(csv_path=Path(RAW_DATASET_PATH))
