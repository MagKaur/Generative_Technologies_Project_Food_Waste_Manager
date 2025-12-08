# Food Waste Manager – RAG System

System do zarządzania przepisami i ograniczania marnowania żywności, oparty o grafową bazę Neo4j, embeddingi wektorowe i modele Azure OpenAI.

---

## 1. Overview i architektura

Aplikacja służy do pobierania przepisów z różnych źródeł (tekst teraz, docelowo również URL/PDF/obraz), parsowania ich przy pomocy LLM do postaci strukturalnej oraz zapisywania w bazie grafowej Neo4j wraz z embeddingami tekstu dla wyszukiwania semantycznego. System można wykorzystać jako backend pod asystenta kuchennego w stylu RAG: wyszukuje przepisy po składnikach, podobieństwie semantycznym oraz planowanej spiżarni użytkownika.

Wysoki poziom przepływu danych:
1. Wejście (tekst przepisu, CSV, plik) trafia do `IngestionService`.  
2. `LLMClient` wywołuje Azure OpenAI: najpierw ekstrakcja struktury przepisu, potem generowanie embeddingów fragmentów tekstu.  
3. `Neo4jClient` zapisuje węzły i relacje w grafie (Recipe, Ingredient, Document, Chunk, User, PantryItem itd.), w tym indeks wektorowy dla embeddingów.  

---

## 2. Technologie i struktura projektu

**Stos technologiczny:**  
- Python 3.12 (obraz bazowy w `Dockerfile`).  
- Neo4j (kontener `neo4j` w `docker-compose.yaml`).  
- Azure OpenAI (chat model + embeddings przez `langchain_openai`).  
- LangChain (kompozycja promptów i structured output).  

**Struktura katalogów (logiczna):**  
- `src/config/config.py` – ładowanie `.env`, konfiguracja DB, Azure OpenAI, logowanie.  
- `src/database/db_client.py` – klient Neo4j, inicjalizacja constraintów/indeksów, metody upsert/query.  
- `src/database/init_loader.py` – inicjalizacja bazy z pliku CSV (bulk ingest przepisów).  
- `src/models/graph_models.py` – dataclassy domenowe odwzorowujące węzły i relacje w grafie (Recipe, Ingredient, User, Chunk itd.).  
- `src/models/llm_models.py` – schematy Pydantic na structured output z LLM (`RecipeSchema`, `IngredientSchema`).  
- `src/services/llm_client.py` – warstwa komunikacji z Azure OpenAI (parsowanie przepisu, embeddingi, chunking).  
- `src/services/ingestion_service.py` – pipeline ingestion (Input → Recipe → Document+Chunks → Neo4j).  
- `src/main.py` – prosty entrypoint pokazujący ingest pojedynczego pliku tekstowego.  

Zależności Pythona (fragment): `neo4j`, `python-dotenv`, `langchain_core`, `langchain_openai`, `uuid_utils` są zdefiniowane w `requirements.txt` i instalowane w obrazie Dockera.

---

## 3. Konfiguracja i zmienne środowiskowe

Konfiguracja aplikacji opiera się na zmiennych środowiskowych ładowanych przez `python-dotenv` z pliku `.env` w katalogu `src/config` (ścieżka jest zakodowana w `config.py`).

**Kluczowe zmienne:**
- Baza Neo4j:  
  - `DB_URI` (np. `neo4j://neo4j:7687` w Dockerze lub `bolt://localhost:7687` lokalnie).  
  - `DB_USER`, `DB_PASSWORD`, `DB_DATABASE`.  
- Dane wejściowe:  
  - `RAW_DATASET_PATH` – ścieżka do źródłowego CSV z przepisami używana przez `init_loader.py`.  
  - `DATASET_PATH` – alternatywna ścieżka na potrzeby innych pipeline’ów (np. przetworzony CSV).  
- Azure OpenAI:  
  - `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`.  
  - `AZURE_OPENAI_MODEL` (np. `gpt-4.1-nano`), `AZURE_OPENAI_EMBEDDING_MODEL` (np. `-embedding-3-small`).  
  - `OPENAI_API_VERSION`.  
- Logowanie: `LOG_LEVEL`, opcjonalnie `LOG_FILE`.  

Przykładowy minimalny `.env` znajduje się w env.example (wartości należy podstawić własne):



---

## 4. Uruchomienie (lokalnie i w Dockerze)

### 4.1. Lokalne uruchomienie (Python)

1. Zainstaluj zależności:
pip install -r requirements.txt


Następnie uruchom Neo4j lokalnie (np. przez oficjalny obraz Dockera) i ustaw `DB_URI`, `DB_USER`, `DB_PASSWORD` na pasujące do instancji.

2. Upewnij się, że `.env` zawiera poprawne dane logowania do Neo4j i dane Azure OpenAI.

3. Ingest pojedynczego pliku tekstowego (np. `./data/gofry.txt`):
python src/main.py


Skrypt:
- Inicjalizuje bazę (`init_database()` – drop wszystkiego + odtworzenie constraintów/indeksów).  
- Wczytuje plik tekstowy, przekazuje go do `IngestionService.ingest_recipe()`, a następnie zamyka połączenie z DB.  

4. Inicjalizacja z pliku CSV:
python src/database/init_loader.py


Skrypt:
- Resetuje bazę (`init_database()`).  
- Przechodzi po wierszach CSV z `RAW_DATASET_PATH`, łączy kolumny w jeden tekst (`row_to_`), i dla każdego wiersza wywołuje ingestion przepisu.  

### 4.2. Uruchomienie za pomocą Docker Compose

Plik `docker-compose.yaml` definiuje dwa serwisy: Neo4j oraz kontener z aplikacją inicjalizującą dane.

1. Uruchom:
docker-compose up --build


- Serwis `neo4j` wystawia porty `7474` (UI) i `7687` (bolt), wykorzystuje wolumen `./neo4j_data:/data` dla trwałości danych.  
- Serwis `init-app` budowany jest z `Dockerfile`, montuje katalog `./src` i `./data` do `/app/src` i `/app/data` oraz startuje poleceniem `PYTHONPATH=/app/src python -m src.main` po zweryfikowaniu dostępności Neo4j.  

2. Zatrzymanie i wyczyszczenie:
docker-compose down # samo zatrzymanie
docker-compose down -v # zatrzymanie + usunięcie danych Neo4j



---

## 5. Model danych, wyszukiwanie i dalszy rozwój

### 5.1. Model grafowy i indeksy

Model domenowy opisany w `graph_models.py` jest mapowany na węzły i relacje Neo4j w metodach `Neo4jClient`. Kluczowe węzły:
- `Recipe`, `Ingredient`, `Cuisine`, `Tag`, `DietaryProfile`, `User`, `PantryItem`, `Document`, `Chunk`, `Season`.  
- Relacje m.in.: `HAS_INGREDIENT`, `OF_CUISINE`, `HAS_TAG`, `SUITABLE_FOR`, `OWNS`, `IS_OF_INGREDIENT`, `HAS_CHUNK`, `DESCRIBES`, `MENTIONS_INGREDIENT`.  

Przy inicjalizacji tworzone są:
- Constrainty unikalności dla ID/nazw (np. `Recipe.id`, `Ingredient.id`/`name`, `Document.id`, `Chunk.id`, `Cuisine.name`, `Tag.name` itd.).  
- Indeksy dla zapytań (np. `Recipe.title`, `Recipe.total_time_minutes`, `Ingredient.name`, `PantryItem.expiration_date`).  
- Wektorowy indeks `chunk_embedding_index` na polu `Chunk.embedding` (1536 wymiarów, cosinus), używany w `find_similar_chunks()` do semantycznego wyszukiwania fragmentów tekstu.  

### 5.2. Pipeline ingestion i rola LLM

`IngestionService` realizuje główny pipeline:  
1. `_extract_()` normalizuje wejście do tekstu (aktualnie tylko `IngestionSource.`, inne typy rzucają `NotImplementedError`).  
2. `LLMClient.parse_recipe_to_domain()` wysyła prompt systemowy do Azure OpenAI z opisem schematu `RecipeSchema` i zasadami ekstrakcji (tłumaczenie na EN, ekstrakcja wszystkich składników z całego tekstu, inferencja ilości, tagów itd.).  
3. Zwrócony `Recipe` jest zapisywany do Neo4j przez `upsert_recipe()`.  
4. Surowy tekst dzielony jest na chunki przez `simple_chunk()`, dla każdego chunku generowany jest embedding oraz lista składników wspomnianych w tym fragmencie (`filter_ingredients_for_chunk()`).  
5. Tworzony jest `Document` z listą `Chunk` i zapisywany przez `upsert_document_with_chunks()` z relacjami `HAS_CHUNK`, `DESCRIBES`, `MENTIONS_INGREDIENT`.  

