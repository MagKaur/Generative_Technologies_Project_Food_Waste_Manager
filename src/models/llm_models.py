from pydantic import BaseModel
from typing import List, Optional


class IngredientSchema(BaseModel):
    name: str  # Unique in graph
    amount: float  # For HasIngredientRel
    unit: str  # For HasIngredientRel
    category: str
    in_seasons: List[str]  # Names for Season connect


class RecipeSchema(BaseModel):
    title: str  # Unique in graph
    total_time_minutes: Optional[int] = None
    cuisine: Optional[str] = None  # Name for Cuisine
    tags: Optional[List[str]] = None  # Names for Tag
    dietary_profiles: Optional[List[str]] = None  # Names for DietaryProfile
    ingredients: List[IngredientSchema]
    instructions: Optional[str] = None


class MatchItem(BaseModel):
    input_index: int
    matched_name: Optional[str] = None  # For Ingredient by name
    confidence: float


class MatchSchema(BaseModel):
    matches: List[MatchItem]
