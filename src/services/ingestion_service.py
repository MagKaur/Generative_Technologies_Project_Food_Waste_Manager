from dataclasses import dataclass
from enum import Enum
from typing import Optional
from uuid import uuid4

from src.database.db_client import Neo4jClient
from src.services.llm_client import LLMClient, simple_chunk
from src.models.graph_models import Recipe, Document, Chunk, Ingredient


class IngestionSource(str, Enum):
    TEXT = "text"
    URL = "url"
    PDF = "pdf"
    IMAGE = "image"


@dataclass
class IngestionInput:
    source: IngestionSource
    content: str  # text, URL, file path etc.
    raw_bytes: Optional[bytes] = None  # PDF/image


@dataclass
class IngestionResult:
    recipe_id: str
    document_id: str


class IngestionService:
    def __init__(self, db_client: Neo4jClient, llm_client: LLMClient):
        self._db = db_client
        self._llm = llm_client

    def ingest_recipe(self, data: IngestionInput) -> IngestionResult:
        """
        Main method: accepts different input types, get raw text from recipe
        parses to Recipe and inserts to database.
        """
        raw_text = self._extract_text(data)

        # 1) text -> Recipe
        recipe: Recipe = self._llm.parse_recipe_to_domain(raw_text)

        # 2) Recipe saving
        self._db.upsert_recipe(recipe)

        # 3) Build Document + Chunk
        document_id = str(uuid4())

        chunks_list = []
        for position, chunk_text in enumerate(simple_chunk(raw_text)):
            chunk_ingredients = filter_ingredients_for_chunk(chunk_text, recipe.ingredients)
            ch = Chunk(
                id=str(uuid4()),
                text=chunk_text,
                embedding=self._llm.embed_text(chunk_text),
                position=position,
                ingredients=chunk_ingredients,
                document_id=document_id,
            )
            chunks_list.append(ch)

        document = Document(
            id=document_id,
            source_type=data.source.value,
            raw_text=raw_text,
            recipe=recipe,
            chunks=chunks_list,
        )

        # 4) Save Document + Chunk
        self._db.upsert_document_with_chunks(document)

        return IngestionResult(recipe_id=recipe.id, document_id=document.id)

    # --- PRIVATE HELPING METHODS ---

    def _extract_text(self, data: IngestionInput) -> str:
        """ Normalize input to raw recipe text. Adapters for HTML, PDF, IMAGE etc..."""
        if data.source == IngestionSource.TEXT:
            return data.content

        if data.source == IngestionSource.URL:
            # TODO: download HTML and extract recipe text
            raise NotImplementedError("URL ingestion not implemented yet.")

        if data.source == IngestionSource.PDF:
            # TODO: pdf -> text (ex. pdfminer / pypdf)
            raise NotImplementedError("PDF ingestion not implemented yet.")

        if data.source == IngestionSource.IMAGE:
            # TODO: OCR -> text
            raise NotImplementedError("Image ingestion not implemented yet.")

        raise ValueError(f"Unsupported ingestion source: {data.source}")


def filter_ingredients_for_chunk(chunk_text: str, ingredients: list[Ingredient]) -> list[Ingredient]:
    lower = chunk_text.lower()
    return [ing for ing in ingredients if ing.name.lower() in lower]