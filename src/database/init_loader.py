import csv
import datetime
import logging
from pathlib import Path

from src.services.ingestion_service import IngestionService, IngestionInput, IngestionSource, UserInput, PantryItemInput
from src.database.db_client import DatabaseService  # Adapted to neomodel client
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


def init_recipes(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"File {csv_path} not found")

    db = DatabaseService(uri=DB_URI, user=DB_USER, password=DB_PASSWORD, database=DB_DATABASE)

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

                result = ingestion.ingest_recipe(input_)
                print(result)

                if i % 20 == 0:
                    logger.info("Loaded %d recipes...", i)
            except Exception as e:
                logger.exception("Error in line %d: %s", i, e)

    logger.info("Init DB with recipes from CSV successfully ended.")


def init_user_pantry_item(users_csv_path: Path, items_csv_path: Path):
    if not users_csv_path.exists():
        raise FileNotFoundError(f"File {users_csv_path} not found")
    if not items_csv_path.exists():
        raise FileNotFoundError(f"File {items_csv_path} not found")

    db = DatabaseService(uri=DB_URI, user=DB_USER, password=DB_PASSWORD, database=DB_DATABASE)
    llm = LLMClient()
    ingestion = IngestionService(db_client=db, llm_client=llm)

    user_data = None
    with users_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            try:
                print(row)
                # Adapted: dietary_profiles split by comma, email=None (not in CSV)
                user_data = UserInput(
                    name=row["name"],
                    email=row["email"],
                    dietary_profiles=[p.strip() for p in row["dietary_profiles"].split(",") if p.strip()],
                )

                if i % 20 == 0:
                    logger.info("Loaded %d users...", i)
            except Exception as e:
                logger.exception("Error in line %d: %s", i, e)

    items_to_add = []
    with items_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            try:
                item = PantryItemInput(
                    name=row["ingredient_name"],
                    quantity=float(row["quantity"]),  # Ensure float
                    unit=row["unit"],
                    expiration_date=datetime.date.fromisoformat(row["expiration_date"]) if row[
                        "expiration_date"] else None,
                )
                items_to_add.append(item)

                if i % 20 == 0:
                    logger.info("Loaded %d pantry items...", i)
            except Exception as e:
                logger.exception("Error in line %d: %s", i, e)

    # Set pantry_items and ingest user (neomodel handles upsert/connect)
    user_data.pantry_items = items_to_add
    user_id = ingestion.ingest_user(user_data)  # New method, returns id

    logger.info(
        f"Init DB with user {user_data.name} and {len(items_to_add)} pantry items from CSV successfully ended. User ID: {user_id}")


if __name__ == "__main__":
    db = DatabaseService(uri=DB_URI, user=DB_USER, password=DB_PASSWORD, database=DB_DATABASE)
    check = db.get_all_ingredients()

    if not check:
        print("No ingredients found. Initializing database ...")
        init_user_pantry_item(Path(RAW_DATASET_PATH) / "users.csv", Path(RAW_DATASET_PATH) / "pantry_items.csv")
        init_recipes(csv_path=Path(RAW_DATASET_PATH) / "recipes_extracted_full_version.csv")
    else:
        print("Already initialized. Skipping.")