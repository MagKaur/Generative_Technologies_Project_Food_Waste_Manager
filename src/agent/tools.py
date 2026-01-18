# src/agent/tools.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.services.ingestion_service import (
    IngestionService,
    IngestionInput,
    IngestionSource,
    PantryItemInput,
    UserInput,
)
from src.database.database_service import DatabaseService


class ToolName(str, Enum):
    # Ingest recipes
    INGEST_RECIPE_TEXT = "ingest_recipe_text"
    INGEST_RECIPE_URL = "ingest_recipe_url"
    INGEST_RECIPE_PDF = "ingest_recipe_pdf"
    INGEST_RECIPE_IMAGE = "ingest_recipe_image"

    # Pantry / user
    CREATE_USER = "create_user"
    ADD_PANTRY_ITEMS = "add_pantry_items"
    SHOW_PANTRY = "show_pantry"

    # Recipes search / filters
    SEARCH_RECIPES = "search_recipes"
    SEARCH_FROM_PANTRY = "search_recipes_from_pantry"
    SEARCH_FROM_LIST = "search_recipes_from_list"
    SEARCH_EXPIRING = "search_recipes_expiring"
    SEARCH_UNDER_TIME = "search_recipes_under_time"

    # Shopping list
    MISSING_INGREDIENTS = "get_missing_ingredients"

    # RAG
    RAG_SEARCH = "rag_search"

    # Compositions
    PLAN_COURSES = "plan_courses_for_guests"
    SEASONAL_CUISINE = "seasonal_cuisine_query"


@dataclass
class ToolResult:
    type: str              # for frontend rendering, e.g. "recipes_list"
    message: str           # short human-friendly summary
    data: Dict[str, Any]   # payload


class AgentTools:
    """
    Thin tool layer: wraps your existing services into agent-callable functions.
    Keeps agent/orchestrator clean.
    """

    def __init__(self, db: DatabaseService, ingestion: IngestionService):
        self.db = db
        self.ingestion = ingestion

    # -------------------------
    # A) INGEST RECIPE
    # -------------------------
    def ingest_recipe_text(self, text: str) -> ToolResult:
        res = self.ingestion.ingest_recipe(
            IngestionInput(source=IngestionSource.TEXT, content=text)
        )
        return ToolResult(
            type="recipe_ingested",
            message=f"Dodałam przepis: {res.recipe_id}",
            data={"recipe_id": res.recipe_id, "document_id": res.document_id},
        )

    def ingest_recipe_url(self, url: str) -> ToolResult:
        res = self.ingestion.ingest_recipe(
            IngestionInput(source=IngestionSource.URL, content=url)
        )
        return ToolResult(
            type="recipe_ingested",
            message=f"Dodałam przepis z linku: {res.recipe_id}",
            data={"recipe_id": res.recipe_id, "document_id": res.document_id, "url": url},
        )

    def ingest_recipe_pdf(self, pdf_bytes: Optional[bytes] = None, pdf_path: Optional[str] = None) -> ToolResult:
        # In your ingestion service, PDF supports raw_bytes (and content can be a path).
        if pdf_bytes is None and not pdf_path:
            raise ValueError("Provide pdf_bytes or pdf_path.")
        res = self.ingestion.ingest_recipe(
            IngestionInput(source=IngestionSource.PDF, content=pdf_path or "uploaded.pdf", raw_bytes=pdf_bytes)
        )
        return ToolResult(
            type="recipe_ingested",
            message=f"Dodałam przepis z PDF: {res.recipe_id}",
            data={"recipe_id": res.recipe_id, "document_id": res.document_id},
        )

    def ingest_recipe_image(self, image_bytes: Optional[bytes] = None, image_path: Optional[str] = None) -> ToolResult:
        if image_bytes is None and not image_path:
            raise ValueError("Provide image_bytes or image_path.")
        res = self.ingestion.ingest_recipe(
            IngestionInput(source=IngestionSource.IMAGE, content=image_path or "uploaded.png", raw_bytes=image_bytes)
        )
        return ToolResult(
            type="recipe_ingested",
            message=f"Dodałam przepis ze zdjęcia: {res.recipe_id}",
            data={"recipe_id": res.recipe_id, "document_id": res.document_id},
        )

    # -------------------------
    # B) USER / PANTRY
    # -------------------------
    def create_user(self, name: str, email: Optional[str] = None, dietary_profiles: Optional[List[str]] = None) -> ToolResult:
        uid = self.ingestion.ingest_user(
            UserInput(name=name, email=email, dietary_profiles=dietary_profiles)
        )
        return ToolResult(
            type="user_created",
            message="Utworzyłam użytkownika.",
            data={"user_id": uid, "name": name, "email": email, "dietary_profiles": dietary_profiles or []},
        )

    def add_pantry_items(self, user_id: str, items: List[Dict[str, Any]]) -> ToolResult:
        """
        items: [{name, quantity, unit, expiration_date?}] expiration_date as ISO string yyyy-mm-dd or None
        """
        user = self.db.get_user_by_id(user_id)
        if not user:
            raise ValueError(f"User not found: {user_id}")

        parsed_items: List[PantryItemInput] = []
        for it in items:
            exp = it.get("expiration_date")
            exp_date = date.fromisoformat(exp) if exp else None
            parsed_items.append(
                PantryItemInput(
                    name=str(it["name"]),
                    quantity=float(it.get("quantity", 0.0)),
                    unit=str(it.get("unit", "")),
                    expiration_date=exp_date,
                )
            )

        # This method does LLM matching and writes to DB via upsert_user_with_pantry
        self.ingestion.ingest_pantry_items_for_user(user, parsed_items)

        pantry = self.db.get_user_pantry(user_id)
        return ToolResult(
            type="pantry_updated",
            message=f"Dodałam {len(parsed_items)} produktów do spiżarni.",
            data={"user_id": user_id, "pantry": pantry},
        )

    def show_pantry(self, user_id: str) -> ToolResult:
        pantry = self.db.get_user_pantry(user_id)
        return ToolResult(
            type="pantry_list",
            message=f"Spiżarnia: {len(pantry)} pozycji.",
            data={"user_id": user_id, "pantry": pantry},
        )

    # -------------------------
    # C) SEARCH RECIPES (core)
    # -------------------------
    def search_recipes(
        self,
        user_id: Optional[str] = None,
        max_minutes: Optional[int] = None,
        required_dietary_profiles: Optional[List[str]] = None,
        excluded_tags: Optional[List[str]] = None,
        include_ingredients: Optional[List[str]] = None,
        include_all_ingredients: bool = False,
        use_pantry_ingredients: bool = False,
        use_expiring_from_pantry: bool = False,
        expiring_days: int = 3,
        course: Optional[str] = None,
        limit: int = 20,
        require_any_ingredient_match: bool = True,
    ) -> ToolResult:
        include_list = include_ingredients or []
        if (len(include_list) == 0) and (not use_pantry_ingredients) and (not use_expiring_from_pantry):
            require_any_ingredient_match = False
        recipes = self.db.search_recipes(
            user_id=user_id,
            max_minutes=max_minutes,
            required_dietary_profiles=required_dietary_profiles,
            excluded_tags=excluded_tags,
            include_ingredients=include_ingredients,
            include_all_ingredients=include_all_ingredients,
            use_pantry_ingredients=use_pantry_ingredients,
            use_expiring_from_pantry=use_expiring_from_pantry,
            expiring_days=expiring_days,
            require_any_ingredient_match=require_any_ingredient_match,
            course=course,
            limit=limit,
        )
        return ToolResult(
            type="recipes_list",
            message=f"Znalazłam {len(recipes)} przepisów.",
            data={"recipes": recipes},
        )

    def search_recipes_from_pantry(self, user_id: str, **kwargs) -> ToolResult:
        return self.search_recipes(user_id=user_id, use_pantry_ingredients=True, include_ingredients=None, **kwargs)

    def search_recipes_from_list(self, ingredients: List[str], **kwargs) -> ToolResult:
        return self.search_recipes(include_ingredients=ingredients, use_pantry_ingredients=False, **kwargs)

    def search_recipes_expiring(self, user_id: str, days: int = 3, **kwargs) -> ToolResult:
        return self.search_recipes(user_id=user_id, use_expiring_from_pantry=True, expiring_days=days, **kwargs)

    def search_recipes_under_time(self, minutes: int, limit: int = 20) -> ToolResult:
        recipes = self.db.find_recipes_under_time(minutes=minutes, limit=limit)
        return ToolResult(
            type="recipes_list",
            message=f"Przepisy do {minutes} minut: {len(recipes)} wyników.",
            data={"recipes": recipes, "max_minutes": minutes},
        )

    # -------------------------
    # D) SHOPPING LIST
    # -------------------------
    def get_missing_ingredients(self, user_id: str, recipe_id_or_title: str) -> ToolResult:
        missing = self.db.get_missing_ingredients_for_recipe(user_id, recipe_id_or_title)
        return ToolResult(
            type="missing_ingredients",
            message=f"Lista zakupów do: {missing.get('title', recipe_id_or_title)}",
            data=missing,
        )

    # -------------------------
    # E) RAG
    # -------------------------
    def rag_search(self, query_text: str, k: int = 5) -> ToolResult:
        # vector_search expects embedding vector; we can embed via ingestion.llm (available there)
        emb = self.ingestion._llm.embed_text(query_text)  # pragmatic; if you prefer, inject llm separately
        chunks = self.db.vector_search_chunks(emb, k=k)
        return ToolResult(
            type="rag_chunks",
            message=f"Znalazłam {len(chunks)} pasujących fragmentów.",
            data={"chunks": chunks, "k": k},
        )

    # -------------------------
    # F) COMPOSITIONS (multi-tool)
    # -------------------------
    def plan_courses_for_guests(
        self,
        user_id: Optional[str] = None,
        include_ingredients: Optional[List[str]] = None,
        max_minutes: Optional[int] = None,
        required_dietary_profiles: Optional[List[str]] = None,
        excluded_tags: Optional[List[str]] = None,
        use_pantry_ingredients: bool = False,
        use_expiring_from_pantry: bool = False,
        expiring_days: int = 3,
        limit_each: int = 10,
    ) -> ToolResult:
        appetizer = self.db.search_recipes(
            user_id=user_id,
            max_minutes=max_minutes,
            required_dietary_profiles=required_dietary_profiles,
            excluded_tags=excluded_tags,
            include_ingredients=include_ingredients,
            use_pantry_ingredients=use_pantry_ingredients,
            use_expiring_from_pantry=use_expiring_from_pantry,
            expiring_days=expiring_days,
            require_any_ingredient_match=True,
            course="appetizer",
            limit=limit_each,
        )
        main = self.db.search_recipes(
            user_id=user_id,
            max_minutes=max_minutes,
            required_dietary_profiles=required_dietary_profiles,
            excluded_tags=excluded_tags,
            include_ingredients=include_ingredients,
            use_pantry_ingredients=use_pantry_ingredients,
            use_expiring_from_pantry=use_expiring_from_pantry,
            expiring_days=expiring_days,
            require_any_ingredient_match=True,
            course="main",
            limit=limit_each,
        )
        dessert = self.db.search_recipes(
            user_id=user_id,
            max_minutes=max_minutes,
            required_dietary_profiles=required_dietary_profiles,
            excluded_tags=excluded_tags,
            include_ingredients=include_ingredients,
            use_pantry_ingredients=use_pantry_ingredients,
            use_expiring_from_pantry=use_expiring_from_pantry,
            expiring_days=expiring_days,
            require_any_ingredient_match=True,
            course="dessert",
            limit=limit_each,
        )
        return ToolResult(
            type="course_plan",
            message="Propozycje na przystawkę, danie główne i deser.",
            data={"appetizer": appetizer, "main": main, "dessert": dessert},
        )

    def seasonal_cuisine_query(
        self,
        cuisine: str,
        season: str,
        user_id: Optional[str] = None,
        max_minutes: Optional[int] = None,
        required_dietary_profiles: Optional[List[str]] = None,
        excluded_tags: Optional[List[str]] = None,
        limit: int = 20,
    ) -> ToolResult:
        """
        MVP stub:
        - We don't have explicit cuisine/season filters in DatabaseService.search_recipes.
        - So we approximate cuisine via tags (e.g., 'indian') and season via ingredient list produced by LLM in the orchestrator.
        TODO: Add a dedicated cypher using (r)-[:OF_CUISINE]->(Cuisine) and Ingredient-Season relations.
        """
        approx_tags = [cuisine.lower()]
        recipes = self.db.search_recipes(
            user_id=user_id,
            max_minutes=max_minutes,
            required_dietary_profiles=required_dietary_profiles,
            excluded_tags=excluded_tags,
            include_ingredients=None,  # let orchestrator pass seasonal ingredients if extracted
            include_all_ingredients=False,
            use_pantry_ingredients=False,
            use_expiring_from_pantry=False,
            require_any_ingredient_match=False,
            course=None,
            limit=limit,
        )
        # Filter client-side by tag hit if tags exist in results
        filtered = [r for r in recipes if any(t == cuisine.lower() for t in (r.get("tags") or []))]
        return ToolResult(
            type="recipes_list",
            message=f"Wyniki dla kuchni={cuisine}, sezon={season} (MVP przybliżenie).",
            data={"recipes": filtered, "cuisine": cuisine, "season": season, "note": "MVP approximation; add DB cypher for cuisine/season for full accuracy."},
        )
