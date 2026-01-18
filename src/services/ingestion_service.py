import logging
import os
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional, List

import pdfplumber
import requests
import easyocr
import cv2
import numpy as np

from bs4 import BeautifulSoup
from neomodel import DoesNotExist

from src.database.database_service import DatabaseService  # Changed to neomodel client
from src.models.graph_models import (
    Recipe, Document, Chunk, Ingredient, User, PantryItem
)
from src.models.llm_models import MatchItem
from src.clients.llm_client import LLMClient
from src.utils.chunking import simple_chunk


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


@dataclass
class UserInput:
    """Input data for ingesting a new user."""
    name: str
    email: Optional[str] = None  # Added for User model
    dietary_profiles: Optional[List[str]] = None  # Names for DietaryProfile
    pantry_items: Optional[List["PantryItemInput"]] = None  # Optional list


@dataclass
class PantryItemInput:
    """Input data for a single pantry item."""
    name: str  # Name of the ingredient (e.g., "tomato" or "pomidor")
    quantity: float
    unit: str  # e.g., "cups", "grams"
    expiration_date: Optional[date] = None


class IngestionService:
    def __init__(self, db_client: DatabaseService, llm_client: LLMClient):  # Changed to DatabaseService
        self._db = db_client
        self._llm = llm_client
        self._match_threshold = 0.7  # Threshold for accepting LLM match

        self._logger = logging.getLogger(__name__)

    def ingest_recipe(self, data: IngestionInput) -> IngestionResult:
        """
        Main method: accepts different input types, get raw text from recipe
        parses to Recipe and inserts to database.
        """
        raw_text = self._extract_text(data)

        # 1) text -> Recipe (with temp data for rels)
        recipe: Recipe = self._llm.parse_recipe_to_domain(raw_text)

        # 2) Recipe saving (neomodel: save + connect with rel properties)
        self._db.upsert_recipe(recipe)

        # 3) Build Document + Chunk
        chunks_list = []
        for position, chunk_text in enumerate(simple_chunk(raw_text)):
            temp_data = []
            if hasattr(recipe, '_ingredients_data'):
                temp_data = recipe._ingredients_data

            chunk_ingredients = filter_ingredients_for_chunk(chunk_text, temp_data)

            ch = Chunk(
                text=chunk_text,
                embedding=self._llm.embed_text(chunk_text),
                position=position,
            )
            ch._ingredients = [data['ingredient'] for data in chunk_ingredients]  # From temp
            chunks_list.append(ch)

        document = Document(
            source_type=data.source.value,
            raw_text=raw_text,
        )
        document._chunks = chunks_list
        document._recipe = recipe

        # 4) Save Document + Chunk
        self._db.upsert_document_with_chunks(document)

        return IngestionResult(recipe_id=recipe.title, document_id=document.uuid)  # Use title as id (unique)

    def ingest_user(self, user_data: UserInput) -> str:
        """
        Ingests a new user with dietary_profiles and optional pantry_items.
        Returns generated user uuid (unique).
        """
        # Create User without assigning relationship lists (connect in db)
        user = User(
            name=user_data.name,
            email=user_data.email,
        )
        if user_data.dietary_profiles:
            user._dietary_profiles_names = user_data.dietary_profiles
        if user_data.pantry_items:
            self.ingest_pantry_items_for_user(user, user_data.pantry_items)

        self._db.upsert_user_with_pantry(user)
        return user.uuid

    def ingest_pantry_items_for_user(self, user: User, pantry_items_data: List[PantryItemInput]) -> None:
        """
        Ingests pantry items for user (by object, not id).
        Uses LLM to match ingredient names to existing (by name).
        Creates new Ingredient if no match or low confidence.
        Builds pairs (pantry_item, ingredient) for db connect (no assignment to rel).
        """
        # 1) Fetch all existing ingredients from DB (by name)
        existing_ingredients = self._db.get_all_ingredients()  # List of {'name': str, 'category': str}
        existing_names = [ing["name"] for ing in existing_ingredients]

        # 2) Prepare input for LLM
        new_items_list = [
            {"name": item.name, "index": i}
            for i, item in enumerate(pantry_items_data)
        ]

        # 3) Call LLM for matching
        if existing_names:
            matches = self._llm.match_ingredients_to_existing(existing_names, new_items_list)
        else:
            matches = [MatchItem(input_index=i, matched_name=None, confidence=0.0) for i in
                       range(len(pantry_items_data))]

        # 4) Build PantryItems with matched or new ingredients (by name)
        pantry_pairs = []
        for match in matches:
            input_index = match.input_index
            item_data = pantry_items_data[input_index]
            if match.matched_name and match.confidence >= self._match_threshold:
                # FIXED: Fetch existing via get (not create new instance)
                try:
                    ingredient = Ingredient.nodes.get(name=match.matched_name)
                    # Update category if needed from candidate (optional)
                    candidate = next((ing for ing in existing_ingredients if ing["name"] == match.matched_name), None)
                    if candidate:
                        ingredient.category = candidate.get("category", ingredient.category)
                        ingredient.save()  # Save updated category if changed
                except DoesNotExist:
                    # Fallback: Create new if get fails (edge case)
                    ingredient = _create_new_ingredient(item_data.name)
            else:
                # No match or low confidence: Create new
                ingredient = _create_new_ingredient(item_data.name)

            pantry_item = PantryItem(
                quantity=item_data.quantity,
                unit=item_data.unit,
                expiration_date=item_data.expiration_date,
            )
            # Do NOT assign pantry_item.ingredient = ingredient – connect later
            pantry_pairs.append((pantry_item, ingredient))  # Pair for db processing

        user._pantry_pairs = pantry_pairs
        self._db.upsert_user_with_pantry(user)

    # --- PRIVATE HELPING METHODS ---

    def _extract_text(self, data: IngestionInput) -> str:
        """ Normalize input to raw recipe text. Adapters for HTML, PDF, IMAGE etc..."""
        if data.source == IngestionSource.TEXT:
            return data.content

        if data.source == IngestionSource.URL:
            return self._process_url(data)

        if data.source == IngestionSource.PDF:
            return self._process_pdf(data)

        if data.source == IngestionSource.IMAGE:
            return self._process_image(data)

        raise ValueError(f"Unsupported ingestion source: {data.source}")

    def _process_url(self, data: IngestionInput):
        try:
            response = requests.get(data.content, timeout=5,
                                    headers={'User-Agent': 'Mozilla/5.0 (compatible; RecipeBot/1.0)'})
            response.raise_for_status()
            html_content = response.text

            # 2) Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')

            # 3) Delete non necessary elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()

            # 4) Get main text
            raw_text = soup.body.get_text(separator='\n', strip=True) if soup.body else ""

            self._logger.info(f"Extracted {len(raw_text)} chars from URL: {data.content}")

            return raw_text

        except requests.exceptions.RequestException as e:
            self._logger.warning(f"Error fetching URL {data.content}: {e}; returning empty text.")
            return ""
        except Exception as e:
            self._logger.error(f"Unexpected error in URL extraction for {data.content}: {e}")
            return ""

    def _process_pdf(self, data: IngestionInput):
        try:
            if data.raw_bytes:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                    tmp.write(data.raw_bytes)
                    pdf_path = tmp.name
            else:
                pdf_path = data.content
                if not os.path.exists(pdf_path):
                    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            raw_text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text()
                    if page_text:
                        raw_text += f"\n--- Page {page_num} ---\n{page_text}\n"

                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            raw_text += "\nTABLE:\n" + "\n".join(
                                [" | ".join(row) for row in table if row]) + "\n"

            if data.raw_bytes:
                os.unlink(pdf_path)

            self._logger.info(f"Extracted text {len(raw_text)} chars from PDF: {pdf_path or 'bytes'}")
            self._logger.debug(f"Extracted text:\n{raw_text}")
            return raw_text.strip()

        except FileNotFoundError as e:
            self._logger.warning(f"PDF file not found: {e}")
            return ""
        except Exception as e:
            self._logger.error(f"Error extracting PDF {data.content}: {e}")
            return ""

    def _process_image(self, data: IngestionInput):
        try:
            reader = easyocr.Reader(['pl', 'en'])
            raw_text = ""

            if data.raw_bytes:
                nparr = np.frombuffer(data.raw_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Could not decode image from bytes")
                result = reader.readtext(image, detail=0, paragraph=True)
                raw_text = '\n'.join(result)
                self._logger.info(f"Extracted {len(raw_text)} chars from image bytes")
            else:
                image_path = data.content
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                result = reader.readtext(image_path, detail=0, paragraph=True)
                raw_text = '\n'.join(result)
                self._logger.info(f"Extracted {len(raw_text)} chars from image: {image_path}")

            self._logger.debug(f"Extracted text:\n{raw_text.strip()}")
            return raw_text.strip()

        except FileNotFoundError as e:
            self._logger.warning(f"Image file not found: {e}")
            return ""
        except Exception as e:
            self._logger.error(f"Error extracting image {data.content}: {e}")
            return ""


def _create_new_ingredient(name: str) -> Ingredient:
    """Helper to create a new Ingredient object with defaults (no id, unique by name)."""
    return Ingredient(
        name=name,
        category="other",  # Default; infer later if needed
    )


def filter_ingredients_for_chunk(chunk_text: str, ingredients_data: list[dict]) -> list[dict]:
    """Filter ingredients for chunk (adjusted for temp data with ingredient obj)."""
    lower = chunk_text.lower()
    return [data for data in ingredients_data if data['ingredient'].name.lower() in lower]
