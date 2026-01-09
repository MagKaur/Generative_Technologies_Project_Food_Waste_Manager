from src.database.database_service import DatabaseService
from src.config.config import DB_URI, DB_USER, DB_PASSWORD, DB_DATABASE

def main():
    db = DatabaseService(uri=DB_URI, user=DB_USER, password=DB_PASSWORD, database=DB_DATABASE)

    # 1) Podmień na ISTNIEJĄCE user_id z Twojej bazy
    user_id = "55bed824-3a3d-48ac-98f2-ec2e284b1e24"

    # 2) get_user_by_id
    u = db.get_user_by_id(user_id)
    print("USER found:", bool(u), "uuid:", getattr(u, "uuid", None))

    # 3) Wypisz produkty usera (pantry)
    pantry = db.get_user_pantry(user_id)
    print("PANTRY size:", len(pantry))
    print("PANTRY first 5:")
    print(pantry[:5])

    # 4) Składniki w bazie (kontrola czy coś w ogóle jest)
    ings = db.get_all_ingredients()
    print("ALL INGREDIENTS size:", len(ings))
    print("ALL INGREDIENTS first 10:")
    print(ings[:10])

    # 5) Przepisy z expiring soon (w oparciu o pantry)
    exp = db.find_recipes_expiring_soon(user_id, days=7, limit=10)
    print("RECIPES USING EXPIRING SOON size:", len(exp))
    print("RECIPES USING EXPIRING SOON first 3:")
    print(exp[:3])

    # 6) Przepisy używające pantry (qty>0)
    using_pantry = db.find_recipes_using_user_pantry(user_id, limit=10)
    print("RECIPES USING USER PANTRY size:", len(using_pantry))
    print("RECIPES USING USER PANTRY first 3:")
    print(using_pantry[:3])

    # 7) Przepisy <= X minut
    under_time = db.find_recipes_under_time(minutes=20, limit=10)
    print("RECIPES UNDER TIME size:", len(under_time))
    print("RECIPES UNDER TIME first 3:")
    print(under_time[:3])

    # 8) Przepisy po składnikach (podmień jak chcesz)
    by_ing = db.find_recipes_by_ingredients(names=["tomato", "zucchini"], require_all=False, limit=10)
    print("RECIPES BY INGREDIENTS size:", len(by_ing))
    print("RECIPES BY INGREDIENTS first 3:")
    print(by_ing[:3])

    # 9) Uniwersalne search_recipes (przykład)
    search = db.search_recipes(
        user_id=user_id,
        max_minutes=30,
        required_dietary_profiles=["vegetarian"],
        excluded_tags=["spicy"],
        include_ingredients=["tomato", "zucchini"],
        include_all_ingredients=False,
        use_pantry_ingredients=False,
        use_expiring_from_pantry=False,
        require_any_ingredient_match=True,
        course="appetizer",
        limit=10,
    )
    print("SEARCH_RECIPES size:", len(search))
    print("SEARCH_RECIPES first 3:")
    print(search[:3])

    # 10) get_missing_ingredients_for_recipe
    # Podmień na istniejący tytuł lub uuid przepisu z Twojej bazy
    recipe_id_or_title = "Quick Tomato Zucchini Salad"
    missing = db.get_missing_ingredients_for_recipe(user_id, recipe_id_or_title)
    print("MISSING INGREDIENTS for recipe:", recipe_id_or_title)
    print(missing)

    # 11) vector_search_chunks
    # Tu musisz mieć: (a) index chunk_embedding, (b) jakieś chunki z embeddingiem 512
    # Dajemy dummy embedding, żeby sprawdzić czy query działa (wyniki mogą być losowe/0)
    embedding = [0.0] * 512
    chunks = db.vector_search_chunks(embedding, k=5)
    print("VECTOR SEARCH CHUNKS size:", len(chunks))
    print("VECTOR SEARCH CHUNKS first 3:")
    print(chunks[:3])


if __name__ == "__main__":
    main()