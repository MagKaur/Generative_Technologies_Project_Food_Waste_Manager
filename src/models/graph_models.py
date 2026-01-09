import uuid
import os

from neomodel import (
    StructuredNode,
    StringProperty,
    IntegerProperty,
    FloatProperty,
    DateProperty,
    ArrayProperty,
    RelationshipTo,
    RelationshipFrom,
    One,
    OneOrMore,
    ZeroOrMore,
    StructuredRel,
    ZeroOrOne,
    VectorIndex, EmailProperty
)
# Fallback to dims 512 jeśli nie dostanie z .env
EMBEDDING_DIMS = int(os.getenv("EMBEDDING_DIMS", "512"))


class Cuisine(StructuredNode):
    name = StringProperty(unique_index=True)  # Unique name


class Season(StructuredNode):
    name = StringProperty(unique_index=True)  # Unique name


class Tag(StructuredNode):
    name = StringProperty(unique_index=True)  # Unique name


class DietaryProfile(StructuredNode):
    name = StringProperty(unique_index=True)  # Unique name

    users = RelationshipFrom('User', 'HAS_DIETARY_PROFILE', cardinality=ZeroOrMore)
    suitable_recipes = RelationshipFrom('Recipe', 'SUITABLE_FOR', cardinality=ZeroOrMore)


class Ingredient(StructuredNode):
    name = StringProperty(unique_index=True)
    category = StringProperty(default='other')

    in_seasons = RelationshipTo("Season", 'IN_SEASON', cardinality=ZeroOrMore)
    recipes = RelationshipFrom('Recipe', 'HAS_INGREDIENT', cardinality=ZeroOrMore)
    pantry_items = RelationshipFrom('PantryItem', 'IS_OF_INGREDIENT', cardinality=ZeroOrMore)


class HasIngredientRel(StructuredRel):
    amount = FloatProperty(required=True)
    unit = StringProperty(required=True)


class Recipe(StructuredNode):
    uuid = StringProperty(unique_index=True, default=lambda: str(uuid.uuid4()))
    title = StringProperty(index=True)
    total_time_minutes = IntegerProperty(required=False)
    instructions = StringProperty(required=False)
    source_type = StringProperty(required=False)

    ingredients = RelationshipTo("Ingredient", 'HAS_INGREDIENT', cardinality=ZeroOrMore, model=HasIngredientRel)

    cuisine = RelationshipTo("Cuisine", 'OF_CUISINE', cardinality=ZeroOrOne)
    tags = RelationshipTo("Tag", 'HAS_TAG', cardinality=ZeroOrMore)
    suitable_for = RelationshipTo("DietaryProfile", 'SUITABLE_FOR', cardinality=ZeroOrMore)
    users_cooked = RelationshipFrom('User', 'COOKED', cardinality=ZeroOrMore)
    documents = RelationshipFrom('Document', 'DESCRIBES', cardinality=ZeroOrMore)


class PantryItem(StructuredNode):
    uuid = StringProperty(unique_index=True, default=lambda: str(uuid.uuid4()))
    quantity = FloatProperty()
    unit = StringProperty()
    expiration_date = DateProperty(required=False)

    owner = RelationshipFrom('User', 'OWNS', cardinality=One)  # Odwrotna do pantry_items
    ingredient = RelationshipTo('Ingredient', 'IS_OF_INGREDIENT', cardinality=ZeroOrOne)


class CookedRel(StructuredRel):
    date = DateProperty(required=False)
    rating = FloatProperty()


class User(StructuredNode):
    uuid = StringProperty(unique_index=True, default=lambda: str(uuid.uuid4()))
    name = StringProperty()
    email = EmailProperty()

    pantry_items = RelationshipTo("PantryItem", 'OWNS', cardinality=ZeroOrMore)
    dietary_profiles = RelationshipTo("DietaryProfile", 'HAS_DIETARY_PROFILE', cardinality=ZeroOrMore)

    cooked_recipes = RelationshipTo("Recipe", 'COOKED', cardinality=ZeroOrMore, model=CookedRel)


class Document(StructuredNode):
    uuid = StringProperty(unique_index=True, default=lambda: str(uuid.uuid4()))
    source_type = StringProperty()
    raw_text = StringProperty()

    recipe = RelationshipTo("Recipe", 'DESCRIBES', cardinality=ZeroOrOne)
    chunks = RelationshipTo("Chunk", 'HAS_CHUNK', cardinality=OneOrMore)


class Chunk(StructuredNode):
    uuid = StringProperty(unique_index=True, default=lambda: str(uuid.uuid4()))
    text = StringProperty()
    embedding = ArrayProperty(base_property=FloatProperty(),
                              vector_index=VectorIndex(dimensions=EMBEDDING_DIMS, similarity_function="cosine"))
    position = IntegerProperty()

    document = RelationshipFrom("Document", 'HAS_CHUNK', cardinality=OneOrMore)
    ingredients = RelationshipTo("Ingredient", 'MENTIONS_INGREDIENT', cardinality=ZeroOrMore)
