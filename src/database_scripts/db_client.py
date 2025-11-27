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
        cypher_statements = [
            "CREATE CONSTRAINT recipe_name_unique IF NOT EXISTS FOR (r:Recipe) REQUIRE r.title IS UNIQUE;",
            "CREATE CONSTRAINT product_name_unique IF NOT EXISTS FOR (p:Product) REQUIRE p.name IS UNIQUE;",
        ]
        with self._driver.session(database=self._database) as session:
            for stmt in cypher_statements:
                session.run(stmt)

    def upsert_recipe_with_ingredients(self, row: RecipeRow) -> None:
        """Adds ingredients to the recipe."""
        cypher = """
            MERGE (r:Recipe {title: $title})

            WITH r, $ingredients AS ingredients
            UNWIND ingredients AS ing
            MERGE (p:Product {name: ing.name})
            MERGE (r)-[rel:NEEDS]->(p)
            SET rel.amount = ing.amount,
                rel.unit   = ing.unit
        """
        params = {
            "title": row.title,
            "ingredients": [
                {"name": ing.name, "amount": ing.amount, "unit": ing.unit}
                for ing in row.ingredients
            ],
        }
        with self._driver.session(database=self._database) as session:
            session.run(cypher, params)

    def load_data(self, df: pd.DataFrame) -> None:
        """Loads whole DataFrame to Neo4j."""
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
        with self._driver.session(database=self._database) as session:
            session.run("MATCH (n) DETACH DELETE n")


def df_row_to_recipe(row: pd.Series) -> RecipeRow:
    ings = [
        Ingredient(
            name=ing_dict["name"],
            amount=ing_dict["amount"],
            unit=ing_dict["unit"],
        )
        for ing_dict in row["ingredients"]
    ]
    return RecipeRow(
        title=str(row["title"]),
        ingredients=ings,
    )



