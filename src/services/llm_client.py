import logging

from typing import Optional
from uuid import uuid4

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from src.config.config import AZURE_OPENAI_MODEL, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, OPENAI_API_VERSION, \
    AZURE_OPENAI_EMBEDDING_MODEL
from src.models.graph_models import Recipe, Ingredient, Cuisine, Tag, Season
from src.models.llm_models import RecipeSchema


def simple_chunk(text: str, max_chars: int = 1000) -> list[str]:
    chunks = []
    current = []
    for line in text.splitlines():
        if sum(len(l) for l in current) + len(line) > max_chars and current:
            chunks.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        chunks.append("\n".join(current))
    return chunks


class LLMClient:
    """Client for communication with Azure OpenAI"""

    def __init__(self, api_endpoint: str = None, api_key: str = None, api_version: str = None, base_model: str = None,
                 embeddings_model: str = None,
                 logger: Optional[logging.Logger] = None) -> None:
        self._client = AzureChatOpenAI(
            azure_endpoint=api_endpoint or AZURE_OPENAI_ENDPOINT,
            api_key=api_key or AZURE_OPENAI_API_KEY,
            api_version=api_version or OPENAI_API_VERSION,
            model=base_model or AZURE_OPENAI_MODEL
        )
        self._logger = logger or logging.getLogger(__name__)
        self._embedder = AzureOpenAIEmbeddings(
            azure_endpoint=api_endpoint or AZURE_OPENAI_ENDPOINT,
            api_key=api_key or AZURE_OPENAI_API_KEY,
            api_version=api_version or OPENAI_API_VERSION,
            model=embeddings_model or AZURE_OPENAI_EMBEDDING_MODEL,
        )

    def parse_recipe_to_domain(self, text: str) -> Recipe:
        system_prompt = """
            You are an expert recipe data extractor. Your task is to analyze the provided text (which may contain a recipe description, ingredients list, steps, etc.) and extract structured data in valid JSON format.

            Rules:
            - If the recipe is written in any language other than English, first translate the entire recipe content to English before extraction. Ensure the translation is natural and accurate.
            - Focus only on the recipe content; ignore unrelated text, ads, or metadata.
            - For ingredients: 
              - Scan the ENTIRE text thoroughly for ANY mentions of ingredients, even if not in a dedicated list. Look for contextual references in descriptions, steps, or casual mentions (e.g., "add flour" in a step, "a glass of flour" scattered in the text, or just "flour" mentioned randomly). Treat every food item, spice, or material as a potential ingredient unless it's clearly not (e.g., equipment like "bowl").
              - Parse each unique ingredient into a structured object. If the same ingredient is mentioned multiple times, merge them into one entry (e.g., sum quantities if possible, or use the most complete description).
              - Infer quantities, units, and notes if not explicit (e.g., "2 cups flour" → amount: "2", unit: "cups"; "a handful of herbs" → amount: "1", unit: "handful"; if just "flour" → amount: null, unit: null).
              - If an ingredient has no category (e.g., "vegetable", "dairy") or seasonality ("in_seasons" field) (e.g., "year-round", "summer"), assign reasonable defaults based on common knowledge (e.g., category: "other" if unknown; in_seasons: ["year-round"] if unknown).
              - Always include every detected ingredient, even if quantity is vague or absent – do not skip any.
            - For tags: Extract or infer relevant tags like "vegetarian", "quick", "spicy" if mentioned or implied.
            - For instructions: Break down into a numbered list of clear, sequential instructions saved as string (e.g., "1. Preheat oven to 180C.\n2. Mix flour and eggs.").
            - If total_time_minutes is not specified, estimate it reasonably (e.g., based on steps complexity) or set to null.
            - Output ONLY the JSON object matching the schema below. No additional text, explanations, or wrappers.
            
            JSON Schema:
            {
              "title": "string", // Recipe name or title; required
              "total_time_minutes": "integer|null", // Total preparation/cooking time in minutes; null if unknown
              "cuisine": "string|null", // e.g., "Italian", "Mexican"; null if not specified
              "tags": ["string"]|null, // Array of tags like ["vegetarian", "gluten-free"]; empty array or null if none
              "ingredients": [
                {
                  "name": "string", // Ingredient name (e.g., "tomatoes")
                  "amount": "string|null", // Quantity (e.g., "2 cups"); null if not specified
                  "unit": "string|null", // Unit (e.g., "grams", "pieces"); null if not specified
                  "category": "string", // e.g., "vegetable", "dairy", "other"
                  "in_seasons": ["string"] // e.g., ["year-round"], ["summer", "winter"]
                }
              ], // Empty array if no ingredients
              "instructions": "string|null" // string of step descriptions (e.g., "1. Preheat oven to 180C.\n2. Mix ingredients."); null or empty if none
            }
            
        """
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessage(content=text),
        ])

        structured_llm = self._client.with_structured_output(RecipeSchema)

        chain = prompt | structured_llm
        parsed: RecipeSchema = chain.invoke({"input_text": text})

        return Recipe(
            id=str(uuid4()),
            title=parsed.title,
            total_time_minutes=parsed.total_time_minutes,
            source_type="ingest",
            ingredients=[
                Ingredient(
                    id=str(uuid4()),
                    name=ing.name,
                    category=ing.category,
                    in_seasons=[Season(season) for season in ing.in_seasons]
                )
                for ing in parsed.ingredients
            ],
            cuisine=Cuisine(name=parsed.cuisine) if parsed.cuisine else None,
            tags=[Tag(name=t) for t in parsed.tags],
            instructions=parsed.instructions,
        )

    def embed_text(self, text: str) -> list[float]:
        """Returns single embedding for save to Chunk.embedding."""
        return self._embedder.embed_query(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch for embedding texts."""
        return self._embedder.embed_documents(texts)
