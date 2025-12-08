import logging

from neo4j import GraphDatabase, Driver

from src.models.graph_models import Recipe, User, Document, Ingredient, Cuisine, DietaryProfile, Tag


class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    ## INFRASTRUCTURE
    def _drop_all(self):
        """ Drops all data, constraints and indexes. """
        with self._driver.session(database=self._database) as session:
            # Data delete
            session.run("MATCH (n) DETACH DELETE n")
            logging.debug("All Data deleted.")

            # Constraints delete
            constraints = session.run("SHOW CONSTRAINTS")
            for record in constraints:
                query = f"DROP CONSTRAINT " + record['name']
                session.run(query)
                logging.debug(f"{record['name']} constraint dropped.")

            # Indexes delete
            indexes = session.run("SHOW INDEXES")
            for record in indexes:
                query = f"DROP INDEX " + record['name']
                session.run(query)
                logging.debug(f"{record['name']} constraint dropped.")

    def _create_constraints_indexes(self) -> None:
        """ Creates constraints. """
        logging.info("Creating constraints...")
        cypher_statements = [
            ## Constraints
            # Recipe
            "CREATE CONSTRAINT recipe_id_unique IF NOT EXISTS FOR(r:Recipe) REQUIRE r.id IS UNIQUE;",
            # Ingredient
            "CREATE CONSTRAINT ingredient_id_unique IF NOT EXISTS FOR(i:Ingredient) REQUIRE i.id IS UNIQUE;",
            "CREATE CONSTRAINT ingredient_name_unique IF NOT EXISTS FOR (i:Ingredient) REQUIRE i.name IS UNIQUE",
            # PantryItem
            "CREATE CONSTRAINT pantry_item_id_unique IF NOT EXISTS FOR(p:PantryItem) REQUIRE p.id IS UNIQUE;",
            # User
            "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR(u:User) REQUIRE u.id IS UNIQUE;",
            # DietaryProfile
            "CREATE CONSTRAINT dietary_profile_id_unique IF NOT EXISTS FOR(d:DietaryProfile) REQUIRE d.id IS UNIQUE;",
            # Document
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR(d:Document) REQUIRE d.id IS UNIQUE;",
            # Chunk
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR(c:Chunk) REQUIRE c.id IS UNIQUE;",
            # Cuisine
            "CREATE CONSTRAINT cuisine_name_unique IF NOT EXISTS FOR(c:Cuisine) REQUIRE c.name IS UNIQUE;",
            # Season
            "CREATE CONSTRAINT season_name_unique IF NOT EXISTS FOR(s:Season) REQUIRE s.name IS UNIQUE;",
            # Tag
            "CREATE CONSTRAINT tag_name_unique IF NOT EXISTS FOR(t:Tag) REQUIRE t.name IS UNIQUE;",

            ## Indexes for faster searching
            # Recipes by name
            "CREATE INDEX recipe_title_index IF NOT EXISTS FOR (r:Recipe) ON (r.title);",
            # Filtering in time
            "CREATE INDEX recipe_time_index IF NOT EXISTS FOR(r:Recipe) ON(r.total_time_minutes);",
            # Ingredients by name
            "CREATE INDEX ingredient_name_index IF NOT EXISTS FOR(i:Ingredient) ON(i.name);",
            # Expiration Date of PantryItem
            "CREATE INDEX pantry_expiration_index IF NOT EXISTS FOR(p:PantryItem) ON(p.expiration_date);",

            ## Vector index for Chunk.embeddings
            """
                CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS
                FOR(c:Chunk) ON(c.embedding)
                OPTIONS
                {
                    indexConfig: {
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'
                    }
                };
            """,
        ]
        with self._driver.session(database=self._database) as session:
            for stmt in cypher_statements:
                session.run(stmt)
                logging.debug(f"{stmt} executed.")

    def init_database(self) -> None:
        """ Clears database and creates constraints. """
        self._drop_all()
        self._create_constraints_indexes()

    def close(self) -> None:
        self._driver.close()

    ## QUERIES
    def query(self, cypher: str, params: dict | None = None) -> list[dict]:
        with self._driver.session(database=self._database) as session:
            result = session.run(cypher, params or {})
            return [record.data() for record in result]

    def get_recipe(self, recipe_id: str) -> Recipe | None:
        cypher = """
        MATCH (r:Recipe {id: $id})
        OPTIONAL MATCH (r)-[rel:HAS_INGREDIENT]->(i:Ingredient)
        OPTIONAL MATCH (r)-[:OF_CUISINE]->(c:Cuisine)
        OPTIONAL MATCH (r)-[:HAS_TAG]->(t:Tag)
        OPTIONAL MATCH (r)-[:SUITABLE_FOR]->(d:DietaryProfile)
        RETURN r,
               collect(DISTINCT {i: i, rel: rel}) AS ingredients,
               c,
               collect(DISTINCT t) AS tags,
               collect(DISTINCT d) AS diet_profiles
        """
        records = self.query(cypher, {"id": recipe_id})
        if not records:
            return None
        rec = records[0]
        r = rec["r"]
        ingredients = [
            Ingredient(
                id=ing_rec["i"]["id"],
                name=ing_rec["i"]["name"],
                category=ing_rec["i"].get("category"),
            )
            for ing_rec in rec["ingredients"]
            if ing_rec["i"] is not None
        ]
        cuisine = None
        if rec.get("c"):
            cuisine = Cuisine(name=rec["c"]["name"])
        tags = [Tag(name=t["name"]) for t in rec["tags"]]
        suitable_for = [
            DietaryProfile(id=d["id"], name=d.get("name", d["id"]))
            for d in rec["diet_profiles"]
        ]
        return Recipe(
            id=r["id"],
            title=r["title"],
            total_time_minutes=r.get("total_time_minutes"),
            instructions=r.get("instructions"),
            source_type=r.get("source_type"),
            ingredients=ingredients,
            cuisine=cuisine,
            tags=tags,
            suitable_for=suitable_for,
        )

    def find_similar_chunks(
            self,
            embedding: list[float],
            top_k: int = 10,
            min_score: float = 0.0,
    ) -> list[dict]:
        """Returns list of similar chunks + context (Recipe, Ingredient)"""
        cypher = """
           CALL db.index.vector.queryNodes('chunk_embedding_index', $top_k, $embedding)
           YIELD node AS c, score
           WHERE score >= $min_score
           OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(c)
           OPTIONAL MATCH (d)-[:DESCRIBES]->(r:Recipe)
           OPTIONAL MATCH (c)-[:MENTIONS_INGREDIENT]->(i:Ingredient)
           RETURN c, d, r, collect(DISTINCT i) AS ingredients, score
           ORDER BY score DESC
           """
        return self.query(cypher, {
            "embedding": embedding,
            "top_k": top_k,
            "min_score": min_score,
        })

    ## INSERT
    def upsert_recipe(self, recipe: Recipe) -> None:
        """
        Saves / updates:
        - (Recipe)
        - connected Ingredient + relations HAS_INGREDIENT
        - optionally Cuisine + OF_CUISINE
        - optionally Tag + HAS_TAG
        - optionally DietaryProfile + SUITABLE_FOR
        """
        cypher = """
            MERGE (r:Recipe {id: $id})
            SET  r.title              = $title,
                 r.total_time_minutes = $total_time_minutes,
                 r.source_type        = $source_type
            
            // ingredients
            WITH r, $ingredients AS ingredients, $cuisine AS cuisine, $tags AS tags, $diet_profiles AS diet_profiles
            UNWIND ingredients AS ing
              MERGE (i:Ingredient {name: ing.name})
              ON CREATE SET
                  i.id       = ing.id,
                  i.category = ing.category
              SET i.category = coalesce(i.category, ing.category)
              MERGE (r)-[rel:HAS_INGREDIENT]->(i)
              SET   rel.amount = ing.amount,
                    rel.unit   = ing.unit
            
            // cuisine (optional)
            WITH DISTINCT r, cuisine, tags, diet_profiles
            WHERE cuisine IS NOT NULL
            MERGE (c:Cuisine {name: cuisine})
            MERGE (r)-[:OF_CUISINE]->(c)
            
            // tags (optional)
            WITH DISTINCT r, tags, diet_profiles
            UNWIND tags AS tag_name
            MERGE (t:Tag {name: tag_name})
            MERGE (r)-[:HAS_TAG]->(t)
            
            // diet profiles (optional)
            WITH DISTINCT r, diet_profiles
            UNWIND diet_profiles AS dp_name
            MERGE (d:DietaryProfile {id: dp_name})
            ON CREATE SET d.name = dp_name
            MERGE (r)-[:SUITABLE_FOR]->(d)
        """

        ingredients_payload = [
            {
                "id": ing.id,
                "name": ing.name,
                "category": ing.category,
                "amount": getattr(ing, "amount", None),
                "unit": getattr(ing, "unit", None),
            }
            for ing in recipe.ingredients
        ]

        params = {
            "id": recipe.id,
            "title": recipe.title,
            "total_time_minutes": recipe.total_time_minutes,
            "source_type": recipe.source_type,
            "ingredients": ingredients_payload,
            "cuisine": recipe.cuisine.name if recipe.cuisine else None,
            "tags": [t.name for t in recipe.tags],
            "diet_profiles": [dp.name for dp in recipe.suitable_for],
        }

        with self._driver.session(database=self._database) as session:
            session.run(cypher, params)

    def upsert_user_with_pantry(self, user: User) -> None:
        """
        Saves / updates:
        - (User)
        - (PantryItem) + relations OWNS
        - relations PantryItem -> Ingredient
        - relations HAS_DIETARY_PROFILE
        """

        cypher = """
           MERGE (u:User {id: $user.id})
           SET  u.name   = $user.name,
                u.locale = $user.locale

           // PantryItems
           WITH u, $pantry_items AS pantry_items, $diet_profiles AS diet_profiles
           UNWIND pantry_items AS p_item
             MERGE (p:PantryItem {id: p_item.id})
                SET   p.quantity        = p_item.quantity,
                      p.unit            = p_item.unit,
                      p.expiration_date = p_item.expiration_date,
                      p.name            = coalesce(p.name, p_item.name)
             MERGE (u)-[:OWNS]->(p)
             MERGE (i:Ingredient {id: p_item.ingredient_id})
             MERGE (p)-[:IS_OF_INGREDIENT]->(i)



           // DietProfiles
           WITH DISTINCT u, diet_profiles
           UNWIND diet_profiles AS dp_name
             MERGE (d:DietaryProfile {id: dp_name})
             ON CREATE SET d.name = dp_name
             MERGE (u)-[:HAS_DIETARY_PROFILE]->(d)
           """

        pantry_items_payload = [
            {
                "id": p.id,
                "quantity": p.quantity,
                "unit": p.unit,
                "expiration_date": p.expiration_date.isoformat() if p.expiration_date else None,
                "ingredient_id": p.ingredient.id if p.ingredient else None,
                "name": f"{p.ingredient.name} ({p.quantity} {p.unit})" if p.ingredient else None,
            }
            for p in user.pantry_items
        ]

        params = {
            "user": {
                "id": user.id,
                "name": user.name,
                "locale": user.locale,
            },
            "pantry_items": pantry_items_payload,
            "diet_profiles": [dp.name for dp in user.dietary_profiles],
        }

        with self._driver.session(database=self._database) as session:
            session.run(cypher, params)

    def upsert_document_with_chunks(self, doc: Document) -> None:
        """
        Saves / updates:
        - (Document)
        - connection DESCRIBES -> Recipe (if exists)
        - (Chunk) + HAS_CHUNK
        - MENTIONS_INGREDIENT to Ingredient
        """
        cypher = """
        MERGE (d:Document {id: $id})
        SET  d.source_type = $source_type,
             d.raw_text    = $raw_text

        // optionally connection with Recipe
        WITH d, $recipe_id AS recipe_id, $chunks AS chunks
        CALL {
          WITH d, recipe_id
          WITH d, recipe_id WHERE recipe_id IS NOT NULL
          MATCH (r:Recipe {id: recipe_id})
          MERGE (d)-[:DESCRIBES]->(r)
          RETURN 0 AS _
        }

        // chunks + relations to ingredients
        WITH d, chunks
        UNWIND chunks AS ch
          MERGE (c:Chunk {id: ch.id})
          SET   c.text      = ch.text,
                c.embedding = ch.embedding,
                c.position  = ch.position
          MERGE (d)-[:HAS_CHUNK]->(c)

        // MENTIONS_INGREDIENT – only if ingredient exists
        WITH d, c, ch
        UNWIND ch.ingredient_names AS ing_name
        OPTIONAL MATCH (i:Ingredient {name: ing_name})
        FOREACH (_ IN CASE WHEN i IS NULL THEN [] ELSE [1] END |
          MERGE (c)-[:MENTIONS_INGREDIENT]->(i)
        )
        """

        chunks_payload = [
            {
                "id": ch.id,
                "text": ch.text,
                "embedding": ch.embedding,
                "position": ch.position,
                "ingredient_names": [ing.name for ing in ch.ingredients],
            }
            for ch in doc.chunks
        ]

        params = {
            "id": doc.id,
            "source_type": doc.source_type,
            "raw_text": doc.raw_text,
            "recipe_id": doc.recipe.id if doc.recipe else None,
            "chunks": chunks_payload,
        }

        with self._driver.session(database=self._database) as session:
            session.run(cypher, params)
