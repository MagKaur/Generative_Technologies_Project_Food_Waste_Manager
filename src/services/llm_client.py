import json
import logging

from typing import Optional, List
from uuid import uuid4

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from src.config.config import AZURE_OPENAI_MODEL, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, OPENAI_API_VERSION, \
    AZURE_OPENAI_EMBEDDING_MODEL
from src.models.graph_models import Recipe, Ingredient, Cuisine, Tag, Season, DietaryProfile
from src.models.llm_models import RecipeSchema, MatchSchema, MatchItem


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
        """
        Parses text to Recipe domain object.
        amount/unit parsed from LLM for use in HasIngredientRel (not on Ingredient node).
        Ingredient created without id (unique by name).
        """
        system_prompt = """
            ROLE
            You are an expert level recipe parsing and data extraction system. Your task is to analyze raw, unstructured recipe text and convert it into a fully structured JSON object following the schema provided below.
            
            GENERAL RULES
            You must analyze the entire input text including titles, ingredient lists, instructions and any inline mentions of food or preparation steps. Ignore website metadata and unrelated text unless it contains ingredient information.
            
            If the recipe is written in a language other than English, translate the entire content into natural English before processing. All output must be in English.
            
            You must never return null values. If information is missing or implicit, you must infer the most likely and reasonable value based on cooking knowledge and context.
            
            INGREDIENT EXTRACTION AND NORMALIZATION RULES
            Every edible item mentioned anywhere in the text must be extracted as an ingredient.
            
            Ingredient names must be normalized and database friendly.
            Do not include size, preparation style or descriptive adjectives in the name.
            
            Examples
            large whole chicken becomes chicken
            boneless chicken breasts becomes chicken breast
            chopped onion becomes onion
            
            Quantities and units must always be provided.
            If not explicitly stated, infer the most common and reasonable value used in cooking.
            
            Examples
            If ingredient is chicken with no quantity, assume 1 piece
            If ingredient is salt with no amount, assume 1 teaspoon
            If ingredient is butter with no amount, assume 1 tablespoon
            If ingredient is garlic with no amount, assume 1 clove
            If ingredient is onion with no amount, assume 1 piece
            
            Never leave amount or unit empty.
            
            CATEGORY RULES
            Each ingredient must have a category.
            If not obvious, infer the most likely category such as vegetable, meat, dairy, grain, spice or other.
            
            SEASONALITY RULES
            If seasonality is not explicitly stated, always set in_seasons to year-round.
            
            ABBREVIATION AND UNIT NORMALIZATION
            The system must recognize and normalize cooking abbreviations including but not limited to:
            c or c. equals cup
            tbsp or Tbsp equals tablespoon
            tsp equals teaspoon
            oz equals ounce
            lb equals pound
            pkg or pkg. equals package
            pt equals pint
            qt equals quart
            
            For example
            1 c. milk becomes amount 1 unit cup
            2 Tbsp butter becomes amount 2 unit tablespoon
            
            DIETARY PROFILE DEDUCTION RULES
            Dietary profiles must always be inferred from the ingredient list, even if not explicitly stated.
            
            If the recipe contains no meat, poultry or fish, classify it as vegetarian.
            If the recipe contains no animal-derived products such as meat, fish, dairy, eggs or honey, classify it as vegan.
            If the recipe contains no gluten-containing ingredients such as wheat, barley, rye or standard flour, classify it as gluten-free.
            If the recipe contains no dairy products such as milk, butter, cheese or cream, classify it as dairy-free.
            If multiple dietary profiles apply, include all applicable ones.
            If a dietary profile is clearly violated by any ingredient, it must not be included.
            2. Allowed dietary profile values

            Use only the following values:
            vegetarian
            vegan
            gluten-free
            dairy-free
            contains-meat (used only when applicable)
            
            ⚠️ Do not use ambiguous or redundant terms such as:
            non-vegetarian
            contains animal products
            meat-based
            
            TAG INFERENCE
            Infer relevant tags such as sweet, savory, baked, fried, no-bake, dessert, quick, comfort-food, healthy or similar based on ingredients and preparation method.
            
            TIME ESTIMATION
            If total time is not explicitly provided, estimate it based on cooking technique, number of steps and typical preparation time.
            
            INSTRUCTIONS
            Instructions must be returned as a single string with numbered steps in correct order.
            
            OUTPUT REQUIREMENTS
            Return only valid JSON.
            Do not include explanations, comments or any text outside the JSON structure.
            Never return null values.
            Always provide the most reasonable inferred value when data is missing.
            
            JSON SCHEMA
            {
            "title": "string",
            "total_time_minutes": integer,
            "cuisine": "string",
            "tags": ["string"],
            "dietary_profiles": ["string"],
            "ingredients": [
            {
            "name": "string",
            "amount": "string",
            "unit": "string",
            "category": "string",
            "in_seasons": ["string"]
            }
            ],
            "instructions": "string"
            }
        """
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessage(content=text),
        ])

        structured_llm = self._client.with_structured_output(RecipeSchema)

        chain = prompt | structured_llm
        parsed: RecipeSchema = chain.invoke({"input_text": text})

        # Create Recipe without save (save in db_client)
        recipe = Recipe(
            title=parsed.title,
            total_time_minutes=parsed.total_time_minutes,
            instructions=parsed.instructions,
            source_type="ingest",
        )

        # Ingredients: Create without id (unique by name), amount/unit for rel
        recipe._ingredients_data = []  # Temp storage for rel properties
        for ing in parsed.ingredients:
            ingredient = Ingredient(
                name=ing.name,
                category=ing.category,
            )
            # Store for rel: connect in db_client with HasIngredientRel
            recipe._ingredients_data.append({
                'ingredient': ingredient,
                'amount': ing.amount,
                'unit': ing.unit,
                'seasons_data': ing.in_seasons
            })

        # Cuisine, Tags, DietaryProfiles (create/connect in db_client)
        if parsed.cuisine:
            recipe._cuisine = Cuisine(name=parsed.cuisine)
        if parsed.tags:
            recipe._tags = [Tag(name=t) for t in parsed.tags]
        if parsed.dietary_profiles:
            recipe._dietary_profiles = [DietaryProfile(name=d) for d in parsed.dietary_profiles]

        return recipe

    def match_ingredients_to_existing(self, existing_names: List[str], new_items: List[dict]) -> List[MatchItem]:
        """
        Matches new ingredient names to existing ones using semantic similarity.
        Returns list of MatchItem objects (matched_name for Ingredient by name).
        """
        new_items_json = json.dumps(new_items)
        system_content = """
        You are an ingredient matching expert. Given a list of existing ingredients in the database and new items to add,
        for each new item, find the best matching existing ingredient by name (considering variations, synonyms, or inflections like "tomato" vs "tomatoes" or "pomidor" vs "pomidory").

        Rules:
        - Match based on semantic similarity; if no exact or close match, return null for matched_name.
        - Output ONLY the JSON object matching the schema below. No additional text.

        Existing ingredients: {}
        New items: {}

        JSON Schema:
        {{
          "matches": [
            {{
              "input_index": "integer",
              "matched_name": "string|null",
              "confidence": "float"  // 0.0 to 1.0
            }}
          ]
        }}
        """.format(', '.join(existing_names), new_items_json)

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_content),
            HumanMessage(content="Perform the matching based on the provided data."),
        ])

        structured_llm = self._client.with_structured_output(MatchSchema)

        chain = prompt | structured_llm
        parsed: MatchSchema = chain.invoke({})

        return parsed.matches

    def embed_text(self, text: str) -> list[float]:
        """Returns single embedding for save to Chunk.embedding (512-dim for model)."""
        return self._embedder.embed_query(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch for embedding texts."""
        return self._embedder.embed_documents(texts)
