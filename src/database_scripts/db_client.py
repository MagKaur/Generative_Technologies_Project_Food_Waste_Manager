import logging
from dataclasses import dataclass
from typing import List, Optional, LiteralString
from neo4j import GraphDatabase, Driver
import pandas as pd


@dataclass
class Ingredient:
    name: str
    amount: Optional[float]
    unit: Optional[str]


@dataclass
class RecipeRow:
    title: str
    ingredients: List[Ingredient]
    recipe_embedding: Optional[list[float]] = None


class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    def close(self) -> None:
        self._driver.close()

    def init_database(self) -> None:
        """ Clears database and creates constraints. """
        with self._driver.session(database=self._database) as session:
            session.run("MATCH (n) DETACH DELETE n")
            constraints = session.run("SHOW CONSTRAINTS")
            for record in constraints:
                session.run(f"DROP CONSTRAINT {record['name']}")
            self.create_constraints()

    def create_constraints(self) -> None:
        """ Creates constraints. """
        logging.info("Creating constraints...")
        cypher_statements = [
            "CREATE CONSTRAINT recipe_name_unique IF NOT EXISTS FOR (r:Recipe) REQUIRE r.title IS UNIQUE;",
            "CREATE CONSTRAINT product_name_unique IF NOT EXISTS FOR (p:Product) REQUIRE p.name IS UNIQUE;",
            """
            CREATE VECTOR INDEX recipe_embedding_index IF NOT EXISTS
            FOR (r:Recipe) ON (r.recipe_embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'
            }}
            """,
        ]
        with self._driver.session(database=self._database) as session:
            for stmt in cypher_statements:
                session.run(stmt)

    def upsert_recipe_with_ingredients(self, row: RecipeRow) -> None:
        """Adds ingredients to the recipe."""
        cypher = """
            MERGE (r:Recipe {title: $title})
            SET r.recipe_embedding = $recipe_embedding 

            WITH r, $ingredients AS ingredients
            UNWIND ingredients AS ing
            MERGE (p:Product {name: ing.name})
            MERGE (r)-[rel:NEEDS]->(p)
            SET rel.amount = ing.amount,
                rel.unit   = ing.unit
        """
        params = {
            "title": row.title,
            "recipe_embedding": row.recipe_embedding,
            "ingredients": [
                {"name": ing.name, "amount": ing.amount, "unit": ing.unit}
                for ing in row.ingredients
            ],
        }
        with self._driver.session(database=self._database) as session:
            session.run(cypher, params)

    def load_data(self, df: pd.DataFrame) -> None:
        """Loads whole DataFrame to Neo4j."""
        logging.debug("Inserting data into neo4j...")
        for _, row in df.iterrows():
            recipe = df_row_to_recipe(row)
            self.upsert_recipe_with_ingredients(recipe)

    def query(self, cypher: str, params: dict = None) -> list:
        """Gets data from Neo4j."""
        with self._driver.session(database=self._database) as session:
            result = session.run(cypher, params or {})
            return [record.data() for record in result]

    def clear_database(self) -> None:
        """Clears database (without constraints)"""
        logging.debug("Clearing database...")
        with self._driver.session(database=self._database) as session:
            session.run("MATCH (n) DETACH DELETE n")

    def query_similar_recipes_hybrid(self, query_embedding: list[float], user_ingredients: List[str],
                                     min_matches: int = 2, top_k: int = 10) -> list[dict]:
        """
        1. Vector searching top candidates
        2. Filter: recipes with min ingredients count
        Returns: [{'title': ..., 'score': 0.92, 'matches': 3}, ...]
        """
        # Krok 1: Vector searching
        vector_cypher = """
        CALL db.index.vector.queryNodes('recipe_embedding_index', $top_k * 2, $embedding) 
        YIELD node, score
        RETURN node, score
        ORDER BY score DESC
        """
        candidates = self.query(vector_cypher, {"embedding": query_embedding, "top_k": top_k})

        # Krok 2: Cypher filter
        filtered = []
        for cand in candidates[:top_k * 2]:
            title = cand['node']['title']
            filter_cypher = """
            MATCH (r:Recipe {title: $title})-[:NEEDS]->(p:Product)
            WHERE p.name IN $user_ingredients
            RETURN count(p) AS matches
            """
            matches_data = self.query(filter_cypher, {"title": title, "user_ingredients": user_ingredients})
            if matches_data and matches_data[0]['matches'] >= min_matches:
                filtered.append({
                    'title': title,
                    'score': cand['score'],
                    'matches': matches_data[0]['matches']
                })
            if len(filtered) >= top_k:
                break
        return sorted(filtered, key=lambda x: x['score'], reverse=True)[:top_k]



def df_row_to_recipe(row: pd.Series) -> RecipeRow:
    ings = [
        Ingredient(
            name=ing_dict["name"],
            amount=ing_dict["amount"],
            unit=ing_dict["unit"],
        )
        for ing_dict in row["ingredients"]
    ]
    recipe_embedding = row.get("recipe_embedding", None) if "recipe_embedding" in row else None
    return RecipeRow(
        title=str(row["title"]),
        ingredients=ings,
        recipe_embedding=recipe_embedding,
    )



