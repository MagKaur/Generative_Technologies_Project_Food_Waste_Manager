import logging
import uuid
import os
from collections import defaultdict
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from neomodel import db, config
from neomodel.exceptions import DoesNotExist

from src.models.graph_models import (
    User, Recipe, Document, Chunk, Ingredient, Cuisine, DietaryProfile, Tag,
    PantryItem, Season,
)


class DatabaseService:
    """
    OGM Client for Neo4j using neomodel.

    - CRUD via neomodel API
    - constraints/indexes via db.install_labels
    - vector index for Chunk.embedding via raw Cypher (Neo4j 5.15+)
    """

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        self._logger = logging.getLogger(__name__)

        # Accept:
        # - bolt://host:7687
        # - bolt+s://host:7687
        # - neo4j://host:7687 (routing; neomodel by default uses bolt/neo4j driver behind)
        scheme = "bolt"
        if uri.startswith("bolt+s://"):
            scheme = "bolt+s"
        elif uri.startswith("neo4j+s://"):
            scheme = "neo4j+s"
        elif uri.startswith("neo4j://"):
            scheme = "neo4j"
        elif uri.startswith("bolt://"):
            scheme = "bolt"

        # strip scheme
        host_part = uri.split("://", 1)[1] if "://" in uri else uri

        # NOTE: neomodel DATABASE_URL expects user/pass in URI
        # Typical format:
        # bolt://user:pass@host:7687
        # (database selection is not embedded here; neo4j DB chosen by Neo4j config / session)
        full_uri = f"{scheme}://{user}:{password}@{host_part}"
        config.DATABASE_URL = full_uri

        # Encrypted connection flag
        config.ENCRYPTED_CONNECTION = scheme.endswith("+s")

        # If you rely on multi-db, you typically set it in Neo4j driver settings/session.
        # neomodel does not universally support "/{database}" suffix in DATABASE_URL for all drivers,
        # so we keep it as pure connection string above.
        self.database = database

        self._init_models()

    def _init_models(self) -> None:
        """Creates/validates constraints/indexes via neomodel install_labels()."""
        try:
            for model in (User, Recipe, Document, Chunk, Ingredient, Cuisine, DietaryProfile, Tag, PantryItem, Season):
                db.install_labels(model)
            self._logger.info("Neomodel constraints and indexes created/verified.")
        except Exception as e:
            self._logger.warning(f"Error initializing models: {e}")

    def drop_all(self) -> None:
        """
        Drops all nodes/relationships + constraints + indexes.
        Raw cypher because neomodel doesn't provide full schema drop helpers.
        """

        @db.transaction
        def _drop() -> None:
            db.cypher_query("MATCH (n) DETACH DELETE n")
            self._logger.debug("All data deleted.")

            # Drop constraints
            rows, _ = db.cypher_query("SHOW CONSTRAINTS YIELD name RETURN name")
            for (name,) in rows:
                try:
                    db.cypher_query(f"DROP CONSTRAINT {name}")
                    self._logger.debug(f"Constraint dropped: {name}")
                except Exception as e:
                    self._logger.warning(f"Failed to drop constraint {name}: {e}")

            # Drop indexes (vector index included)
            rows, _ = db.cypher_query("SHOW INDEXES YIELD name RETURN name")
            for (name,) in rows:
                try:
                    db.cypher_query(f"DROP INDEX {name}")
                    self._logger.debug(f"Index dropped: {name}")
                except Exception as e:
                    self._logger.warning(f"Failed to drop index {name}: {e}")

        _drop()

    def create_constraints_indexes(self, embedding_dims: Optional[int] = None) -> None:
        """Creates constraints/indexes (standard via neomodel, vector via raw Cypher)."""
        self._init_models()

        if embedding_dims is None:
            embedding_dims = int(os.getenv('EMBEDDING_DIMS', '512'))

        @db.transaction
        def _create_vector_index() -> None:
            try:
                db.cypher_query(
                    """
                    # Connect recipe -> ingredient with props (avoid duplicate rels)
+                   if existing_recipe.ingredients.is_connected(ing):
    +                    rel = existing_recipe.ingredients.relationship(ing)
    +                    # update relationship properties
    +                    rel.amount = float(amount or 0.0)
    +                    rel.unit = unit or ""
    +                    rel.save()
+                   else:
+                        existing_recipe.ingredients.connect(ing, {"amount": float(amount or 0.0), "unit": unit or ""})
                    """,
                    {"dims": embedding_dims},
                )
                self._logger.info("Vector index for Chunk.embedding created/verified.")
            except Exception as e:
                self._logger.warning(
                    f"Error creating vector index: {e} (requires Neo4j 5.15+ and correct property type)"
                )

        _create_vector_index()

    # -------------------------
    # UPSERTS
    # -------------------------

    @db.transaction
    def upsert_user_with_pantry(self, user: User) -> User:
        """
        Upserts User + DietaryProfiles + PantryItems + Ingredients.
        Uses temporary attributes:
          - user._dietary_profiles_names: List[str]
          - user._pantry_pairs: List[Tuple[PantryItem, Ingredient]]
        """

        # Upsert User by uuid
        try:
            existing_user = User.nodes.get(uuid=user.uuid)
            existing_user.name = user.name
            existing_user.email = user.email
            existing_user.save()
            self._logger.debug(f"Updated User {existing_user.uuid}")
        except DoesNotExist:
            existing_user = user.save()
            self._logger.debug(f"Created User {existing_user.uuid}")

        # DietaryProfiles (unique by name)
        for dp_name in getattr(user, "_dietary_profiles_names", []) or []:
            dp_name = (dp_name or "").strip()
            if not dp_name:
                continue
            try:
                profile = DietaryProfile.nodes.get(name=dp_name)
            except DoesNotExist:
                profile = DietaryProfile(name=dp_name).save()
                self._logger.debug(f"Created DietaryProfile {profile.name}")

            if not existing_user.dietary_profiles.is_connected(profile):
                existing_user.dietary_profiles.connect(profile)

        # Pantry items + ingredient mapping
        pantry_pairs: List[Tuple[PantryItem, Ingredient]] = getattr(user, "_pantry_pairs", []) or []
        for p_item, ingredient in pantry_pairs:
            # PantryItem must have uuid
            if not getattr(p_item, "uuid", None):
                p_item.uuid = str(uuid.uuid4())

            # Upsert PantryItem by uuid
            try:
                existing_p = PantryItem.nodes.get(uuid=p_item.uuid)
                existing_p.quantity = p_item.quantity
                existing_p.unit = p_item.unit
                existing_p.expiration_date = p_item.expiration_date
                existing_p.save()
                self._logger.debug(f"Updated PantryItem {existing_p.uuid}")
            except DoesNotExist:
                existing_p = p_item.save()
                self._logger.debug(f"Created PantryItem {existing_p.uuid}")

            # Upsert Ingredient by name (to avoid duplicates)
            ing_name = (ingredient.name or "").strip()
            if not ing_name:
                self._logger.warning(f"Skipping ingredient with empty name for PantryItem {existing_p.uuid}")
                continue

            try:
                ing = Ingredient.nodes.get(name=ing_name)
            except DoesNotExist:
                # if your model requires category, adjust default here
                ing = Ingredient(name=ing_name, category=getattr(ingredient, "category", "other")).save()
                self._logger.debug(f"Created Ingredient {ing.name}")

            # Connect: (User)-[:OWNS]->(PantryItem)
            if not existing_user.pantry_items.is_connected(existing_p):
                existing_user.pantry_items.connect(existing_p)

            # Connect: (PantryItem)-[:IS_OF_INGREDIENT]->(Ingredient)
            if not existing_p.ingredient.is_connected(ing):
                existing_p.ingredient.connect(ing)

        # cleanup temp
        if hasattr(user, "_pantry_pairs"):
            delattr(user, "_pantry_pairs")
        if hasattr(user, "_dietary_profiles_names"):
            delattr(user, "_dietary_profiles_names")

        # Cooked recipes:
        # Neomodel rel managers are not iterable as plain list reliably; use a temp attr if needed.
        cooked_list = getattr(user, "_cooked_recipes_payload", None)
        if cooked_list:
            for cooked in cooked_list:
                # cooked expected fields: title, date, rating (adjust to your payload)
                title = getattr(cooked, "title", None) or cooked.get("title")
                if not title:
                    continue
                try:
                    recipe = Recipe.nodes.get(title=title)
                except DoesNotExist:
                    recipe = Recipe(title=title).save()

                rel_props = {
                    "date": getattr(cooked, "date", None) or cooked.get("date"),
                    "rating": getattr(cooked, "rating", None) or cooked.get("rating"),
                }
                existing_user.cooked_recipes.connect(recipe, rel_props)

        return existing_user

    @db.transaction
    def upsert_recipe(self, recipe: Recipe) -> Recipe:
        """
        Upserts Recipe + cuisine + tags + dietary profiles + ingredients (+ seasons per ingredient).
        Expects temporary fields:
          - recipe._cuisine: Cuisine
          - recipe._tags: List[Tag]
          - recipe._dietary_profiles: List[DietaryProfile]
          - recipe._ingredients_data: List[dict] with keys:
                ingredient (Ingredient), amount, unit, seasons_data (List[str])
        """

        try:
            existing_recipe = Recipe.nodes.get(title=recipe.title)
            existing_recipe.total_time_minutes = recipe.total_time_minutes
            existing_recipe.instructions = recipe.instructions
            existing_recipe.source_type = recipe.source_type
            existing_recipe.save()
            self._logger.debug(f"Updated Recipe {existing_recipe.title}")
        except DoesNotExist:
            existing_recipe = recipe.save()
            self._logger.debug(f"Created Recipe {existing_recipe.title}")

        # Cuisine
        cuisine_obj = getattr(recipe, "_cuisine", None)
        if cuisine_obj:
            try:
                cuisine = Cuisine.nodes.get(name=cuisine_obj.name)
            except DoesNotExist:
                cuisine = cuisine_obj.save()
                self._logger.debug(f"Created Cuisine {cuisine.name}")

            if not existing_recipe.cuisine.is_connected(cuisine):
                existing_recipe.cuisine.connect(cuisine)
            delattr(recipe, "_cuisine")

        # Tags
        tags = getattr(recipe, "_tags", None)
        if tags:
            for tag_obj in tags:
                try:
                    t = Tag.nodes.get(name=tag_obj.name)
                except DoesNotExist:
                    t = tag_obj.save()
                    self._logger.debug(f"Created Tag {t.name}")

                if not existing_recipe.tags.is_connected(t):
                    existing_recipe.tags.connect(t)
            delattr(recipe, "_tags")

        # Dietary profiles
        dps = getattr(recipe, "_dietary_profiles", None)
        if dps:
            for dp_obj in dps:
                try:
                    dp = DietaryProfile.nodes.get(name=dp_obj.name)
                except DoesNotExist:
                    dp = dp_obj.save()
                    self._logger.debug(f"Created DietaryProfile {dp.name}")

                if not existing_recipe.suitable_for.is_connected(dp):
                    existing_recipe.suitable_for.connect(dp)
            delattr(recipe, "_dietary_profiles")

        # Ingredients + seasons
        ing_data_list = getattr(recipe, "_ingredients_data", None)
        if ing_data_list:
            for ing_data in ing_data_list:
                ing_obj: Ingredient = ing_data["ingredient"]
                amount = ing_data.get("amount")
                unit = ing_data.get("unit")
                seasons: List[str] = ing_data.get("seasons_data") or []

                ing_name = (ing_obj.name or "").strip()
                if not ing_name:
                    continue

                try:
                    ing = Ingredient.nodes.get(name=ing_name)
                except DoesNotExist:
                    ing = Ingredient(
                        name=ing_name,
                        category=getattr(ing_obj, "category", "other"),
                    ).save()
                    self._logger.debug(f"Created Ingredient {ing.name}")

                # Connect recipe -> ingredient with props
                existing_recipe.ingredients.connect(ing, {"amount": amount, "unit": unit})

                # seasons on ingredient
                for season_name in seasons:
                    season_name = (season_name or "").strip()
                    if not season_name:
                        continue
                    try:
                        season = Season.nodes.get(name=season_name)
                    except DoesNotExist:
                        season = Season(name=season_name).save()
                        self._logger.debug(f"Created Season {season.name}")

                    if not ing.in_seasons.is_connected(season):
                        ing.in_seasons.connect(season)

            delattr(recipe, "_ingredients_data")

        return existing_recipe

    @db.transaction
    def upsert_document_with_chunks(self, doc: Document) -> Document:
        """
        Upserts Document with Chunks and relationships:
          - (Document)-[:DESCRIBES]->(Recipe)  (if doc._recipe provided)
          - (Document)-[:HAS_CHUNK]->(Chunk)
          - (Chunk)-[:MENTIONS_INGREDIENT]->(Ingredient)
        """

        try:
            existing_doc = Document.nodes.get(uuid=doc.uuid)
            existing_doc.source_type = doc.source_type
            existing_doc.raw_text = doc.raw_text
            existing_doc.save()
            self._logger.debug(f"Updated Document {existing_doc.uuid}")
        except DoesNotExist:
            existing_doc = doc.save()
            self._logger.debug(f"Created Document {existing_doc.uuid}")

        # Optional recipe link
        recipe_obj = getattr(doc, "_recipe", None)
        if recipe_obj:
            # try match by uuid else save
            rec_uuid = getattr(recipe_obj, "uuid", None)
            if rec_uuid:
                try:
                    recipe = Recipe.nodes.get(uuid=rec_uuid)
                except DoesNotExist:
                    recipe = recipe_obj.save()
            else:
                recipe = recipe_obj.save()

            if not existing_doc.recipe.is_connected(recipe):
                existing_doc.recipe.connect(recipe)

        # Chunks
        chunks = getattr(doc, "_chunks", None) or []
        for chunk in chunks:
            try:
                existing_chunk = Chunk.nodes.get(uuid=chunk.uuid)
                existing_chunk.text = chunk.text
                existing_chunk.embedding = chunk.embedding
                existing_chunk.position = chunk.position
                existing_chunk.save()
            except DoesNotExist:
                existing_chunk = chunk.save()

            if not existing_doc.chunks.is_connected(existing_chunk):
                existing_doc.chunks.connect(existing_chunk)

            # Ingredient mentions
            ingredients = getattr(chunk, "_ingredients", None) or []
            for ing_obj in ingredients:
                ing_name = (ing_obj.name or "").strip()
                if not ing_name:
                    continue

                try:
                    ing = Ingredient.nodes.get(name=ing_name)
                except DoesNotExist:
                    try:
                        ing = Ingredient(name=ing_name, category="other").save()
                        self._logger.info(f"Created Ingredient {ing_name} for Chunk {existing_chunk.uuid}")
                    except Exception as e:
                        self._logger.warning(f"Failed to create Ingredient {ing_name}: {e}; skipping.")
                        continue

                if not existing_chunk.ingredients.is_connected(ing):
                    existing_chunk.ingredients.connect(ing)

        existing_doc.save()
        return existing_doc

    # -------------------------
    # READ/DELETE
    # -------------------------

    def get_all_ingredients(self) -> List[Dict[str, Any]]:
        return [{"name": ing.name, "category": ing.category} for ing in Ingredient.nodes.all()]

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        try:
            return User.nodes.get(uuid=user_id)
        except DoesNotExist:
            return None

    def delete_user(self, user: User) -> bool:
        @db.transaction
        def _delete() -> bool:
            try:
                user.delete()
                self._logger.debug(f"Deleted User {user.uuid}")
                return True
            except Exception as e:
                self._logger.error(f"Error deleting User: {e}")
                return False

        return _delete()

    # -------------------------
    # SEARCH / QUERIES
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
        expiring_only_if_qty_gt0: bool = True,
        require_any_ingredient_match: bool = False,
        course: Optional[str] = None,
        course_tag_map: Optional[Dict[str, List[str]]] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        required_dietary_profiles = [
            p.strip().lower() for p in (required_dietary_profiles or []) if p and p.strip()
        ]
        excluded_tags = [t.strip().lower() for t in (excluded_tags or []) if t and t.strip()]
        include_ingredients = [i.strip().lower() for i in (include_ingredients or []) if i and i.strip()]

        course_tag_map = course_tag_map or {
            "appetizer": ["appetizer", "starter", "snack"],
            "main": ["main", "main course", "entree", "dinner"],
            "dessert": ["dessert", "sweet", "cake", "cookie"],
        }
        course = course.strip().lower() if course else None
        course_tags = [t.lower() for t in course_tag_map.get(course, [])] if course else []

        today = date.today()
        end_date = today + timedelta(days=expiring_days)

        query = """
            // 0) include names z czatu
            WITH $include_ingredients AS include_names

            // 1) pantry qty>0 (opcjonalnie) — subquery ZAWSZE zwraca 1 wiersz
            CALL {
              WITH include_names
              OPTIONAL MATCH (u:User {uuid: $user_id})-[:OWNS]->(p:PantryItem)-[:IS_OF_INGREDIENT]->(ing:Ingredient)
              WHERE $use_pantry_ingredients = true
                AND $user_id IS NOT NULL
                AND p.quantity IS NOT NULL AND p.quantity > 0
              RETURN coalesce(collect(DISTINCT toLower(ing.name)), []) AS pantry_names
            }

            // 2) expiring soon (opcjonalnie) — subquery ZAWSZE zwraca 1 wiersz
            CALL {
              WITH include_names
              OPTIONAL MATCH (u:User {uuid: $user_id})-[:OWNS]->(p:PantryItem)-[:IS_OF_INGREDIENT]->(ing:Ingredient)
              WHERE $use_expiring_from_pantry = true
                AND $user_id IS NOT NULL
                AND p.expiration_date IS NOT NULL
                AND p.expiration_date >= $today
                AND p.expiration_date <= $end_date
                AND ($expiring_only_if_qty_gt0 = false OR (p.quantity IS NOT NULL AND p.quantity > 0))
              RETURN coalesce(collect(DISTINCT toLower(ing.name)), []) AS expiring_names
            }

            WITH
              include_names,
              coalesce(pantry_names, []) AS pantry_names,
              coalesce(expiring_names, []) AS expiring_names

            // 3) zbuduj target_names = include + pantry + expiring (unikalne, bez APOC)
            WITH include_names + pantry_names + expiring_names AS all_names
            WITH [x IN all_names WHERE x IS NOT NULL AND x <> ""] AS all_names
            WITH reduce(s = [], x IN all_names | CASE WHEN x IN s THEN s ELSE s + x END) AS target_names

            // 4) recipe i filtry "niezależne od składników"
            MATCH (r:Recipe)
            WHERE ($max_minutes IS NULL OR r.total_time_minutes IS NULL OR r.total_time_minutes <= $max_minutes)

            // wymagane profile
            AND (size($required_profiles) = 0 OR all(p IN $required_profiles WHERE EXISTS {
              MATCH (r)-[:SUITABLE_FOR]->(dp:DietaryProfile)
              WHERE toLower(dp.name) = p
            }))

            // excluded tags
            AND (size($excluded_tags) = 0 OR NOT EXISTS {
              MATCH (r)-[:HAS_TAG]->(t:Tag)
              WHERE toLower(t.name) IN $excluded_tags
            })

            // 5) kurs po tagach (opcjonalnie)
            OPTIONAL MATCH (r)-[:HAS_TAG]->(tag:Tag)
            WITH r, target_names, collect(DISTINCT toLower(tag.name)) AS tags
            WHERE ($course_tags = [] OR any(t IN tags WHERE t IN $course_tags))

            // 6) match składników
            OPTIONAL MATCH (r)-[:HAS_INGREDIENT]->(ri:Ingredient)
            WITH r, tags, target_names, collect(DISTINCT toLower(ri.name)) AS recipe_ings
            WITH
              r, tags, target_names, recipe_ings,
              CASE
                WHEN size(target_names) = 0 THEN []
                ELSE [x IN recipe_ings WHERE x IN target_names]
              END AS matched_list,
              CASE
                WHEN size(target_names) = 0 THEN 0
                ELSE size([x IN recipe_ings WHERE x IN target_names])
              END AS matched_count,
              CASE
                WHEN size(target_names) = 0 THEN 0
                ELSE size([x IN target_names WHERE NOT x IN recipe_ings])
              END AS missing_count

            WHERE
              ($require_any_ingredient_match = false OR matched_count > 0)
              AND ($include_all_ingredients = false OR missing_count = 0)

            RETURN
              r.uuid AS recipe_uuid,
              r.title AS title,
              r.total_time_minutes AS total_time_minutes,
              matched_list AS matched_ingredients,
              tags AS tags,
              matched_count AS matched_count,
              missing_count AS missing_count
            ORDER BY
              matched_count DESC,
              coalesce(r.total_time_minutes, 999999) ASC,
              title ASC
            LIMIT $limit
        """

        params = {
            "user_id": user_id,
            "include_ingredients": include_ingredients,
            "include_all_ingredients": include_all_ingredients,
            "use_pantry_ingredients": use_pantry_ingredients,
            "use_expiring_from_pantry": use_expiring_from_pantry,
            "expiring_only_if_qty_gt0": expiring_only_if_qty_gt0,
            "today": today,
            "end_date": end_date,
            "required_profiles": required_dietary_profiles,
            "excluded_tags": excluded_tags,
            "max_minutes": max_minutes,
            "course_tags": course_tags,
            "require_any_ingredient_match": require_any_ingredient_match,
            "limit": limit,
        }

        rows, _ = db.cypher_query(query, params)

        return [
            {
                "recipe_uuid": r[0],
                "title": r[1],
                "total_time_minutes": r[2],
                "matched_ingredients": r[3] or [],
                "tags": r[4] or [],
                "matched_count": r[5],
                "missing_count": r[6],
            }
            for r in rows
        ]

    def find_recipes_under_time(self, minutes: int, limit: int = 50) -> List[Dict[str, Any]]:
        return self.search_recipes(max_minutes=minutes, limit=limit, require_any_ingredient_match=False)

    def find_recipes_by_ingredients(
        self,
        names: List[str],
        require_all: bool = False,
        max_missing: Optional[int] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        names = [n.strip() for n in names if n and n.strip()]
        if not names:
            return []

        include_all = require_all or (max_missing == 0)

        return self.search_recipes(
            include_ingredients=names,
            include_all_ingredients=include_all,
            require_any_ingredient_match=True,
            limit=limit,
        )

    def find_recipes_expiring_soon(self, user_id: str, days: int = 3, limit: int = 50) -> List[Dict[str, Any]]:
        return self.search_recipes(
            user_id=user_id,
            use_expiring_from_pantry=True,
            expiring_days=days,
            expiring_only_if_qty_gt0=True,
            require_any_ingredient_match=True,
            limit=limit,
        )

    def find_recipes_using_user_pantry(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        return self.search_recipes(
            user_id=user_id,
            use_pantry_ingredients=True,
            require_any_ingredient_match=True,
            limit=limit,
        )

    def get_user_pantry(self, user_id: str) -> List[Dict[str, Any]]:
        query = """
        MATCH (u:User {uuid: $user_id})-[:OWNS]->(p:PantryItem)
        OPTIONAL MATCH (p)-[:IS_OF_INGREDIENT]->(i:Ingredient)
        RETURN
          p.uuid AS pantry_item_uuid,
          coalesce(i.name, "") AS ingredient_name,
          p.quantity AS quantity,
          p.unit AS unit,
          p.expiration_date AS expiration_date
        ORDER BY
          (p.expiration_date IS NULL) ASC,
          p.expiration_date ASC,
          ingredient_name ASC
        """
        rows, _ = db.cypher_query(query, {"user_id": user_id})

        return [
            {
                "pantry_item_uuid": r[0],
                "ingredient_name": r[1],
                "quantity": r[2],
                "unit": r[3],
                "expiration_date": r[4],
            }
            for r in rows
        ]

    def get_user_pantry_by_name(self, user_id: str) -> List[Dict[str, Any]]:
        query = """
           MATCH (u:User {name: $user_id})-[:OWNS]->(p:PantryItem)
           OPTIONAL MATCH (p)-[:IS_OF_INGREDIENT]->(i:Ingredient)
           RETURN
             p.uuid AS pantry_item_uuid,
             coalesce(i.name, "") AS ingredient_name,
             p.quantity AS quantity,
             p.unit AS unit,
             p.expiration_date AS expiration_date
           ORDER BY
             (p.expiration_date IS NULL) ASC,
             p.expiration_date ASC,
             ingredient_name ASC
           """
        rows, _ = db.cypher_query(query, {"user_id": user_id})

        return [
            {
                "pantry_item_uuid": r[0],
                "ingredient_name": r[1],
                "quantity": r[2],
                "unit": r[3],
                "expiration_date": r[4],
            }
            for r in rows
        ]

    def get_missing_ingredients_for_recipe(self, user_id: str, recipe_id_or_title: str) -> Dict[str, Any]:
        query = """
        MATCH (r:Recipe)
        WHERE r.uuid = $rid OR r.title = $rid

        MATCH (r)-[rel:HAS_INGREDIENT]->(ri:Ingredient)
        WITH r, collect({
          name: ri.name,
          amount: rel.amount,
          unit: rel.unit
        }) AS recipe_ings

        MATCH (u:User {uuid: $user_id})-[:OWNS]->(:PantryItem)-[:IS_OF_INGREDIENT]->(pi:Ingredient)
        WITH r, recipe_ings, collect(DISTINCT pi.name) AS pantry_names

        WITH
          r,
          pantry_names,
          recipe_ings,
          [x IN recipe_ings WHERE x.name IN pantry_names] AS owned,
          [x IN recipe_ings WHERE NOT x.name IN pantry_names] AS missing

        RETURN
          r.uuid AS recipe_uuid,
          r.title AS title,
          r.total_time_minutes AS total_time_minutes,
          owned AS owned_ingredients,
          missing AS missing_ingredients
        """
        rows, _ = db.cypher_query(query, {"user_id": user_id, "rid": recipe_id_or_title})

        if not rows:
            return {
                "recipe_uuid": None,
                "title": None,
                "total_time_minutes": None,
                "owned_ingredients": [],
                "missing_ingredients": [],
            }

        r = rows[0]
        return {
            "recipe_uuid": r[0],
            "title": r[1],
            "total_time_minutes": r[2],
            "owned_ingredients": r[3] or [],
            "missing_ingredients": r[4] or [],
        }

    #traditional RAG (chunks only R=retrival)
    def vector_search_chunks(self, query_embedding: List[float], k: int = 8) -> List[Dict[str, Any]]:
        cypher = """
        CALL db.index.vector.queryNodes('chunk_embedding', $k, $embedding)
        YIELD node, score

        OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(node)
        OPTIONAL MATCH (d)-[:DESCRIBES]->(r:Recipe)

        RETURN
          node.uuid AS chunk_uuid,
          node.text AS text,
          score AS score,
          d.uuid AS document_uuid,
          r.uuid AS recipe_uuid,
          r.title AS recipe_title,
          r.total_time_minutes AS total_time_minutes
        ORDER BY score DESC
        """
        rows, _ = db.cypher_query(cypher, {"k": k, "embedding": query_embedding})

        return [
            {
                "chunk_uuid": r[0],
                "text": r[1],
                "score": r[2],
                "document_uuid": r[3],
                "recipe_uuid": r[4],
                "recipe_title": r[5],
                "total_time_minutes": r[6],
            }
            for r in rows
        ]

    #traditional RAG (aggregation results method)
    def rag_search_recipes(
            self,
            query_embedding: List[float],
            k_chunks: int = 30,
            limit_recipes: int = 5,
            chunks_per_recipe: int = 3,
            max_minutes: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Classic RAG (baseline) – vector-based retrieval over chunks,
        aggregated to the recipe level.

        Steps:
        1) vector_search_chunks(query_embedding, k_chunks) -> list of chunks
        2) group chunks by recipe_uuid
        3) rank recipes by the best (maximum) chunk similarity score
        4) return top recipes with top chunks as textual context
        5) optionally filter recipes by max_minutes
           (possible because total_time_minutes is now included)
        """

        # Retrieve top-k chunks using vector similarity
        hits = self.vector_search_chunks(query_embedding=query_embedding, k=k_chunks)

        # 1) Group chunks by recipe
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for h in hits:
            recipe_id = h.get("recipe_uuid")
            if not recipe_id:
                continue
            grouped[recipe_id].append(h)

        # 2) Build recipe-level results
        results: List[Dict[str, Any]] = []

        for recipe_id, recipe_hits in grouped.items():
            # Sort chunks within a recipe by similarity score (descending)
            recipe_hits_sorted = sorted(
                recipe_hits,
                key=lambda x: (x.get("score") is None, -(x.get("score") or 0.0))
            )

            best_chunk = recipe_hits_sorted[0]
            total_time = best_chunk.get("total_time_minutes")

            # 3) Optional time filter
            if max_minutes is not None and total_time is not None and total_time > max_minutes:
                continue

            # 4) Recipe score = best chunk similarity score
            best_score = best_chunk.get("score") or 0.0

            results.append(
                {
                    "recipe_uuid": recipe_id,
                    "title": best_chunk.get("recipe_title"),
                    "total_time_minutes": total_time,
                    "rag_best_score": best_score,
                    "rag_top_chunks": [
                        {
                            "chunk_uuid": c.get("chunk_uuid"),
                            "score": c.get("score"),
                            "text": c.get("text"),
                        }
                        for c in recipe_hits_sorted[:chunks_per_recipe]
                    ],
                }
            )

        # 5) Rank recipes by best similarity score
        results = sorted(
            results,
            key=lambda r: -(r.get("rag_best_score") or 0.0)
        )

        return results[:limit_recipes]