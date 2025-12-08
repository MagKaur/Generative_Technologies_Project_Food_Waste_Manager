from pydantic import BaseModel
from typing import List, Optional

class IngredientSchema(BaseModel):
    name: str
    amount: Optional[float] = None
    unit: Optional[str] = None
    category: Optional[str] = None
    in_seasons: Optional[List[str]] = None

class RecipeSchema(BaseModel):
    title: str
    total_time_minutes: Optional[int] = None
    cuisine: Optional[str] = None
    tags: List[str] = []
    ingredients: List[IngredientSchema]
    instructions: Optional[str] = None