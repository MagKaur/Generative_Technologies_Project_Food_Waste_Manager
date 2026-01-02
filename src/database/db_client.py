import logging
import sys
import uuid
from typing import List, Optional, Dict, Any

from neomodel import (
    db,
    config,
)
from neomodel.exceptions import DoesNotExist

from src.models.graph_models import (
    User, Recipe, Document, Chunk, Ingredient, Cuisine, DietaryProfile, Tag,
    PantryItem, Season, )


class DatabaseService:
    """
    OGM Client for Neo4j using neomodel.
    Modeled after Neo4jClient: Handles CRUD via neomodel API.
    Automatically manages constraints/indexes via install_labels.
    For vector index in Chunk: Uses raw Cypher (neomodel lacks native support).
    """

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        # Configure neomodel connection
        full_uri = f"bolt://{user}:{password}@{uri.split('://')[1]}/{database}"
        config.DATABASE_URL = full_uri
        config.ENCRYPTED_CONNECTION = uri.startswith("bolt+s")  # For secure connections
        self._logger = logging.getLogger(__name__)
        self._init_models()  # Automatically create constraints on first use

    def _init_models(self) -> None:
        """Initializes models: Creates constraints/indexes via neomodel."""
        try:
            db.install_labels([
                User, Recipe, Document, Chunk, Ingredient, Cuisine,
                DietaryProfile, Tag, PantryItem,
            ], quiet=False, stdout=sys.stdout)
            self._logger.info("Neomodel constraints and indexes created/verified.")
        except Exception as e:
            self._logger.warning(f"Error initializing models: {e}")

    def _drop_all(self) -> None:
        """Drops all data, constraints and indexes (raw Cypher, as neomodel doesn't have built-in support)."""

        # Use decorator for transaction
        @db.transaction
        def _drop_data():
            # Data delete (DETACH for relationships)
            db.cypher_query("MATCH (n) DETACH DELETE n")
            self._logger.debug("All data deleted.")

            # Constraints delete
            constraints = db.cypher_query("SHOW CONSTRAINTS")
            for record in constraints:
                db.cypher_query(f"DROP CONSTRAINT {record['name']}")
                self._logger.debug(f"Constraint {record['name']} dropped.")

            # Indexes delete
            indexes = db.cypher_query("SHOW INDEXES")
            for record in indexes:
                db.cypher_query(f"DROP INDEX {record['name']}")
                self._logger.debug(f"Index {record['name']} dropped.")

        _drop_data()

    def _create_constraints_indexes(self) -> None:
        """Creates constraints/indexes (neomodel handles automatically, but for vector index – raw)."""
        self._init_models()  # Use neomodel's install_labels for standard ones

        # Vector index for Chunk.embedding (Neo4j 5.15+; raw Cypher)
        @db.transaction
        def _create_vector_index():
            try:
                db.cypher("""
                    CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
                    FOR (c:Chunk) ON (c.embedding)
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: 512,  // Adjusted for model (text-embedding-3-small)
                        `vector.similarity_function`: 'cosine'
                    }}
                """)
                self._logger.info("Vector index for Chunk.embedding created.")
            except Exception as e:
                self._logger.warning(f"Error creating vector index: {e} (requires Neo4j 5.15+)")

        _create_vector_index()

    @db.transaction
    def upsert_user_with_pantry(self, user: User) -> User:
        """
        Upserts User with PantryItems, DietaryProfiles and COOKED relationships.
        Creates new PantryItem/Ingredient if needed (via get_or_create).
        Modeled after the original: Handles relationships with properties.
        """
        # Upsert User (by uuid – assuming model has uuid)
        try:
            existing_user = User.nodes.get(uuid=user.uuid)  # Adjust to your field (uuid or id)
            existing_user.name = user.name
            existing_user.email = user.email
            existing_user.save()
            self._logger.debug(f"Updated User {user.uuid}")
        except DoesNotExist:
            existing_user = user.save()
            self._logger.debug(f"Created User {user.uuid}")

        # DietaryProfiles (upsert by name, connect)
        dietary_profiles_to_process = getattr(user, '_dietary_profiles_names', [])  # Temp str list or empty
        for dp in dietary_profiles_to_process:
            try:
                existing_profile = DietaryProfile.nodes.get(name=dp)
            except DoesNotExist:
                new_profile = DietaryProfile(name=dp)
                existing_profile = new_profile.save()
                self._logger.debug(f"Created DietaryProfile {new_profile.name}")
            existing_user.dietary_profiles.connect(existing_profile)

        # PantryItems + Ingredients (upsert by id/name, connect)
        pantry_pairs = getattr(user, '_pantry_pairs', [])  # Temp list of (pantry_item, ingredient) or empty
        for p_item, ingredient in pantry_pairs:  # FIXED: Iterate over pairs
            # FIXED: Generate uuid if not set (for new)
            if not hasattr(p_item, 'uuid') or p_item.uuid is None:
                p_item.uuid = str(uuid.uuid4())

            # Upsert PantryItem (by uuid)
            try:
                existing_p = PantryItem.nodes.get(uuid=p_item.uuid)
                # Update if exists
                existing_p.quantity = p_item.quantity
                existing_p.unit = p_item.unit
                existing_p.expiration_date = p_item.expiration_date
                existing_p.save()
                self._logger.debug(f"Updated PantryItem {p_item.uuid}")
            except DoesNotExist:
                # Create new
                existing_p = p_item.save()
                self._logger.debug(f"Created new PantryItem {p_item.uuid}")

            # FIXED: Save ingredient if new (check if saved)
            try:
                # If ingredient has uuid, it's saved; else save
                if hasattr(ingredient, 'uuid') and ingredient.uuid:
                    ing = ingredient  # Already saved
                else:
                    ing = ingredient.save()
                    self._logger.debug(f"Saved new Ingredient {ing.name}")
            except DoesNotExist:
                # Fallback if get fails (for matched)
                ing = ingredient.save()

            # FIXED: Connect relationships (explicit, on manager – not on node)
            existing_user.pantry_items.connect(existing_p)
            existing_p.ingredient.connect(ing)  # Manager .connect() – safe
            self._logger.debug(f"Connected PantryItem {p_item.uuid} to Ingredient {ing.name}")

        # Clean temp after processing
        if hasattr(user, '_pantry_pairs'):
            delattr(user, '_pantry_pairs')

        # Cooked Recipes (with properties date/rating)
        for cooked in user.cooked_recipes:
            try:
                recipe = Recipe.nodes.get(title=cooked.title)
            except DoesNotExist:
                recipe = cooked.save()

            # Connect with properties
            existing_user.cooked_recipes.connect(recipe, {'date': cooked.date, 'rating': cooked.rating})

        return existing_user

    @db.transaction
    def upsert_recipe(self, recipe: Recipe) -> Recipe:
        """
        Upserts Recipe with relationships (ingredients with properties, cuisine, tags, suitable_for).
        Handles temp attributes set in parse_recipe_to_domain (_ingredients_data, _cuisine, etc.).
        Connects seasons per ingredient from _ingredients_data.
        Modeled after the original (assuming standard, as truncated).
        """
        try:
            existing_recipe = Recipe.nodes.get(title=recipe.title)
            # Update properties if exists
            existing_recipe.title = recipe.title
            existing_recipe.total_time_minutes = recipe.total_time_minutes
            existing_recipe.instructions = recipe.instructions
            existing_recipe.source_type = recipe.source_type
            existing_recipe.save()
            self._logger.debug(f"Updated Recipe {recipe.title}")
        except DoesNotExist:
            existing_recipe = recipe.save()
            self._logger.debug(f"Created Recipe {recipe.title}")

        if hasattr(recipe, '_cuisine') and recipe._cuisine:
            try:
                cuisine = Cuisine.nodes.get(name=recipe._cuisine.name)
            except DoesNotExist:
                cuisine = recipe._cuisine.save()
                self._logger.debug(f"Created Cuisine {cuisine.name}")
            existing_recipe.cuisine.connect(cuisine)
            delattr(recipe, '_cuisine')  # Clean temp

        if hasattr(recipe, '_tags') and recipe._tags:
            for tag in recipe._tags:
                try:
                    t = Tag.nodes.get(name=tag.name)
                except DoesNotExist:
                    t = tag.save()
                    self._logger.debug(f"Created Tag {t.name}")
                existing_recipe.tags.connect(t)
            delattr(recipe, '_tags')  # Clean temp

        if hasattr(recipe, '_dietary_profiles') and recipe._dietary_profiles:
            for dp in recipe._dietary_profiles:
                try:
                    profile = DietaryProfile.nodes.get(name=dp.name)
                except DoesNotExist:
                    profile = dp.save()
                    self._logger.debug(f"Created DietaryProfile {profile.name}")
                existing_recipe.suitable_for.connect(profile)
            delattr(recipe, '_dietary_profiles')  # Clean temp

        if hasattr(recipe, '_ingredients_data') and recipe._ingredients_data:
            for ing_data in recipe._ingredients_data:
                ing_obj = ing_data['ingredient']  # Ingredient instance from LLM
                amount = ing_data['amount']
                unit = ing_data['unit']
                seasons = ing_data['seasons_data']

                # Upsert Ingredient (by name, unique)
                try:
                    ing = Ingredient.nodes.get(name=ing_obj.name)
                    self._logger.debug(f"Matched to existing Ingredient {ing.name}")
                except DoesNotExist:
                    ing = ing_obj.save()
                    self._logger.debug(f"Created Ingredient {ing.name}")

                # Connect with properties on HasIngredientRel
                existing_recipe.ingredients.connect(ing, {'amount': amount, 'unit': unit})

                for season_name in seasons:
                    try:
                        season = Season.nodes.get(name=season_name)
                    except DoesNotExist:
                        season = Season(name=season_name).save()
                        self._logger.debug(f"Created Season {season.name}")
                    ing.in_seasons.connect(season)

        return existing_recipe

    @db.transaction
    def upsert_document_with_chunks(self, doc: Document) -> Document:
        """
        Upserts Document with Chunks and relationships (DESCRIBES Recipe, HAS_CHUNK, MENTIONS_INGREDIENT).
        Modeled after the original.
        """
        try:
            existing_doc = Document.nodes.get(uuid=doc.uuid)
            existing_doc.source_type = doc.source_type
            existing_doc.raw_text = doc.raw_text
            existing_doc.save()
            self._logger.debug(f"Updated Document {doc.uuid}")
        except DoesNotExist:
            existing_doc = doc.save()
            self._logger.debug(f"Created Document {doc.uuid}")

        # DESCRIBES Recipe (optional)
        recipe_to_process = doc._recipe if hasattr(doc, '_recipe') else None
        if recipe_to_process:
            try:
                existing_recipe = Recipe.nodes.get(uuid=recipe_to_process.uuid)
            except DoesNotExist:
                existing_recipe = recipe_to_process.save()
            existing_doc.save()

            if not existing_doc.recipe.is_connected(existing_recipe):
                existing_doc.recipe.connect(existing_recipe)

        # Chunks
        chunks_to_process = doc._chunks if hasattr(doc, '_chunks') else []
        for chunk in chunks_to_process:
            try:
                existing_chunk = Chunk.nodes.get(uuid=chunk.uuid)
                # Update properties
                existing_chunk.text = chunk.text
                existing_chunk.embedding = chunk.embedding
                existing_chunk.position = chunk.position
                existing_chunk.save()
            except DoesNotExist:
                existing_chunk = chunk.save()

            # Connect HAS_CHUNK
            if not existing_doc.chunks.is_connected(existing_chunk):
                existing_doc.chunks.connect(existing_chunk)

            # MENTIONS_INGREDIENT
            ingredients_to_process = chunk._ingredients if hasattr(chunk, '_ingredients') else []
            for ing in ingredients_to_process:
                ingredient_name = ing.name
                try:
                    ingredient = Ingredient.nodes.get(name=ingredient_name)
                except DoesNotExist:
                    try:
                        new_ingredient = Ingredient(name=ingredient_name, category='other')
                        new_ingredient.save()
                        ingredient = new_ingredient
                        self._logger.info(f"Created Ingredient {ingredient_name} for Chunk {chunk.uuid}")
                    except Exception as e:
                        self._logger.warning(f"Failed to create Ingredient {ingredient_name}: {e}; skipping.")
                        continue
                if not existing_chunk.ingredients.is_connected(ingredient):
                    existing_chunk.ingredients.connect(ingredient)
                    self._logger.debug(f"Connected MENTIONS_INGREDIENT: Chunk {chunk.uuid} -> {ingredient_name}")
                else:
                    self._logger.debug(f"MENTIONS_INGREDIENT already exists: Chunk {chunk.uuid} -> {ingredient_name}")
            existing_chunk.save()

        existing_doc.save()
        return existing_doc

    def get_all_ingredients(self) -> List[Dict[str, Any]]:
        """
        Retrieves all Ingredients (modeled after the original).
        """
        return [
            {
                'name': ing.name,
                'category': ing.category,
            }
            for ing in Ingredient.nodes.all()
        ]

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Read: Get User by ID."""
        try:
            return User.nodes.get(uuid=user_id)
        except DoesNotExist:
            return None

    def delete_user(self, user: User) -> bool:
        """Delete: Remove User and related (cascade via DETACH)."""

        @db.transaction
        def _delete():
            try:
                user.delete()
                self._logger.debug(f"Deleted User {user.uuid}")
                return True
            except Exception as e:
                self._logger.error(f"Error deleting User: {e}")
                return False

        return _delete()
