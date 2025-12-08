from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional


@dataclass
class Ingredient:
    id: str
    name: str
    category: Optional[str] = None

    in_seasons: List["Season"] = field(default_factory=list)  # (Ingredient)-[:IN_SEASON]->(Season)


@dataclass
class Recipe:
    id: str
    title: str
    total_time_minutes: Optional[int] = None
    instructions: Optional[str] = None
    source_type: Optional[str] = None  # np. "kaggle" / "user_chat" / "url" / "image_ocr"

    # relations
    ingredients: List[Ingredient] = field(default_factory=list)  # HAS_INGREDIENT -> Ingredient
    cuisine: Optional["Cuisine"] = None  # OF_CUISINE -> Cuisine
    tags: Optional[List["Tag"]] = None  # HAS_TAG -> Tag
    suitable_for: List["DietaryProfile"] = field(default_factory=list)  # SUITABLE_FOR -> DietaryProfile


@dataclass
class PantryItem:
    id: str
    quantity: float
    unit: str
    expiration_date: Optional[date] = None

    # relations
    ingredient: Optional[Ingredient] = None  # IS_OF_INGREDIENT -> Ingredient
    owner_id: Optional[str] = None  # OWNS <- User.id


@dataclass
class User:
    id: str
    name: str
    locale: Optional[str] = None

    # relations
    pantry_items: Optional[List[PantryItem]] = field(default_factory=list)  # OWNS -> PantryItem
    dietary_profiles: Optional[List["DietaryProfile"]] = field(
        default_factory=list)  # HAS_DIETARY_PROFILE -> DietaryProfile
    cooked_recipes: List[Recipe] = Optional[field(default_factory=list)]  # COOKED -> Recipe


@dataclass
class DietaryProfile:
    id: str
    name: str

    # relations
    users: List[User] = field(default_factory=list)  # HAS_DIETARY_PROFILE <- User
    suitable_recipes: List[Recipe] = field(default_factory=list)  # SUITABLE_FOR <- Recipe
    forbidden_ingredients: List[Ingredient] = field(default_factory=list)  # NOT_ALLOWED_FOR <- Ingredient


@dataclass
class Document:
    id: str
    source_type: str  # np. "kaggle", "url", "user_chat", "image_ocr"
    raw_text: str

    # relations
    recipe: Optional[Recipe] = None  # DESCRIBES -> Recipe
    chunks: List["Chunk"] = field(default_factory=list)  # HAS_CHUNK -> Chunk


@dataclass
class Chunk:
    id: str
    text: str
    embedding: Optional[list[float]] = None
    position: Optional[int] = None

    # relations
    ingredients: List[Ingredient] = field(default_factory=list)  # MENTIONS_INGREDIENT -> Ingredient
    document_id: Optional[str] = None  # HAS_CHUNK <- Document.id


@dataclass
class Cuisine:
    name: str


@dataclass
class Season:
    name: str


@dataclass
class Tag:
    name: str
