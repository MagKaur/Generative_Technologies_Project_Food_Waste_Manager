from __future__ import annotations

import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, cast

import requests
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import AzureOpenAI

from src.eval.deepeval_metrics import (
    RetrievalCase,
    PrecisionAtKMetric,
    RecallAtKMetric,
    MRRAtKMetric,
    NDCGAtKMetric,
)

# =========================
# Env + paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # root projektu
ENV_PATH = PROJECT_ROOT / "src" / "config" / ".env"
load_dotenv(ENV_PATH)

API_URL = os.getenv("EVAL_API_URL", "http://127.0.0.1:8000/agent/message")
TOP_K = int(os.getenv("EVAL_TOP_K", "5"))

_raw_dataset = os.getenv("EVAL_DATASET", "src/eval/datasets/rag_vs_graph.json")
DATASET_PATH = str((PROJECT_ROOT / _raw_dataset).resolve())

print("DB_URI:", os.getenv("DB_URI"))
print("DB_USER:", os.getenv("DB_USER"))
print("DB_PASSWORD set:", bool(os.getenv("DB_PASSWORD")))
print("DATASET_PATH =", DATASET_PATH)
print("DATASET_EXISTS =", Path(DATASET_PATH).exists())


# =========================
# API helpers
# =========================

def call_agent(message: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    payload = {"message": message}
    if user_id:
        payload["user_id"] = user_id


    r = requests.post(API_URL, data=payload, timeout=60)
    r.raise_for_status()
    return cast(Dict[str, Any], r.json())


def extract_ranked_recipe_uuids(resp: Dict[str, Any]) -> List[str]:
    if not isinstance(resp, dict):
        return []

    data = resp.get("data") or {}
    if not isinstance(data, dict):
        return []

    candidates: List[Any] = []
    for key in ("recipes", "results", "items", "ranked", "documents", "context", "chunks"):
        val = data.get(key)
        if isinstance(val, list) and val:
            candidates = val
            break

    if not candidates:
        return []

    out: List[str] = []

    def pick_id(obj: Dict[str, Any]) -> Optional[str]:
        for k in ("recipe_uuid", "uuid", "id", "recipe_id"):
            v = obj.get(k)
            if v:
                return str(v)

        md = obj.get("metadata")
        if isinstance(md, dict):
            for k in ("recipe_uuid", "uuid", "id", "recipe_id"):
                v = md.get(k)
                if v:
                    return str(v)

        doc = obj.get("document")
        if isinstance(doc, dict):
            for k in ("recipe_uuid", "uuid", "id", "recipe_id"):
                v = doc.get(k)
                if v:
                    return str(v)
            md2 = doc.get("metadata")
            if isinstance(md2, dict):
                for k in ("recipe_uuid", "uuid", "id", "recipe_id"):
                    v = md2.get(k)
                    if v:
                        return str(v)

        return None

    for item in candidates:
        if isinstance(item, dict):
            uid = pick_id(item)
            if uid:
                out.append(uid)

    return out


def extract_latency_ms(resp: Dict[str, Any]) -> Optional[float]:
    metrics = (resp.get("data") or {}).get("metrics") or {}
    val = metrics.get("total_ms")
    return float(val) if isinstance(val, (int, float)) else None


# =========================
# Translation (GT alignment)
# =========================

def translate_ingredients_en_singular_via_llm(ingredients: List[str]) -> List[str]:
    """
    Tłumaczy składniki PL->EN i normalizuje do liczby pojedynczej.
    Używa Azure OpenAI – wymagane env:
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_MODEL
    """
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )

    system = """
            TASK
            Translate the provided ingredient names into English and normalize them.
            
            RULES
            - Translate ONLY the ingredient names.
            - Do NOT add, remove, or infer ingredients.
            - Do NOT explain translations.
            - Output MUST be a JSON array of strings.
            - Preserve the original order.
            - If an ingredient is already in English, keep it unchanged.
            - If plural, output SINGULAR in English.
            
            Examples:
            - "pomidory" -> "tomato"
            - "jajka" -> "egg"
            """

    msg = json.dumps(ingredients, ensure_ascii=False)

    messages = cast(Any, [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": msg},
    ])

    resp = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_MODEL"),
        messages=messages,
        temperature=0,
    )

    text = resp.choices[0].message.content.strip()
    return cast(List[str], json.loads(text))


# =========================
# Ground-truth from Neo4j
# =========================

def ground_truth_from_ingredients_via_neo4j(ingredients: List[str]) -> Set[str]:
    uri = os.getenv("NEO4J_URI") or os.getenv("DB_URI")
    user = os.getenv("NEO4J_USER") or os.getenv("DB_USER")
    password = os.getenv("NEO4J_PASSWORD") or os.getenv("DB_PASSWORD")
    db_name = os.getenv("NEO4J_DATABASE") or os.getenv("DB_DATABASE")  # opcjonalne

    if not (uri and user and password):
        raise RuntimeError(
            "Brakuje env: DB_URI / DB_USER / DB_PASSWORD "
            "(lub NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD)"
        )

    ingredients_en = translate_ingredients_en_singular_via_llm(ingredients)
    names = [x.strip().lower() for x in ingredients_en if x and x.strip()]
    if not names:
        return set()

    cypher = """
    MATCH (r:Recipe)-[:HAS_INGREDIENT]->(i:Ingredient)
    WITH r, collect(DISTINCT toLower(i.name)) AS ings
    WHERE all(x IN $names WHERE x IN ings)
    RETURN r.uuid AS recipe_uuid
    """

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        if db_name:
            with driver.session(database=db_name) as session:
                result = session.run(cypher, names=names)
                return {record["recipe_uuid"] for record in result if record.get("recipe_uuid")}
        else:
            with driver.session() as session:
                result = session.run(cypher, names=names)
                return {record["recipe_uuid"] for record in result if record.get("recipe_uuid")}
    finally:
        driver.close()


def get_relevant_set(item: Dict[str, Any]) -> Set[str]:
    if item.get("relevant_recipe_uuids"):
        return set(item["relevant_recipe_uuids"])
    if item.get("ingredients"):
        return ground_truth_from_ingredients_via_neo4j(item["ingredients"])
    raise ValueError(f"Dataset item {item.get('id')} musi mieć `ingredients` albo `relevant_recipe_uuids`.")


# =========================
# DeepEval-only scoring
# =========================

def compute_deepeval_scores(qid: str, query: str, ranked: List[str], relevant: Set[str], k: int) -> Dict[str, float]:
    case = RetrievalCase(qid=qid, query=query, ranked=ranked, relevant=relevant, k=k)
    return {
        f"p@{k}": PrecisionAtKMetric(k).measure(case),
        f"r@{k}": RecallAtKMetric(k).measure(case),
        f"mrr@{k}": MRRAtKMetric(k).measure(case),
        f"ndcg@{k}": NDCGAtKMetric(k).measure(case),
    }


def summarize_latency(vals: List[float]) -> Dict[str, Optional[float]]:
    if not vals:
        return {"p50": None, "p95": None, "mean": None}
    vals_sorted = sorted(vals)
    p50 = statistics.median(vals_sorted)
    p95 = vals_sorted[max(0, int(math.ceil(0.95 * len(vals_sorted))) - 1)]
    mean = sum(vals_sorted) / len(vals_sorted)
    return {"p50": p50, "p95": p95, "mean": mean}


# =========================
# Main eval
# =========================

def run() -> None:
    dataset = json.load(open(DATASET_PATH, "r", encoding="utf-8"))

    results: List[Dict[str, Any]] = []
    latency: Dict[str, List[float]] = {"graph": [], "rag": []}

    for item in dataset:
        qid = item["id"]
        query = item["query"]
        ingredients = item.get("ingredients", [])

        relevant = get_relevant_set(item)

        print(f"\n--- {qid} ---")
        print("ingredients:", ingredients)
        print("relevant_count:", len(relevant))
        print("relevant_sample:", list(relevant)[:5])

        # 1) call both modes (zgodnie z Twoimi promptami)
        resp_graph = call_agent(f"[MODE=GRAPH] {query}")
        resp_rag = call_agent(f"[MODE=RAG] {query}")

        # 2) extract ranked uuids
        ranked_graph = extract_ranked_recipe_uuids(resp_graph)
        ranked_rag = extract_ranked_recipe_uuids(resp_rag)

        print("graph_top5:", ranked_graph[:5])
        print("rag_top5:", ranked_rag[:5])
        print("intersection_graph:", len(set(ranked_graph) & relevant))
        print("intersection_rag:", len(set(ranked_rag) & relevant))

        # 3) DeepEval-only metrics
        graph_scores = compute_deepeval_scores(qid, query, ranked_graph, relevant, TOP_K)
        rag_scores = compute_deepeval_scores(qid, query, ranked_rag, relevant, TOP_K)

        print("DeepEval graph:", graph_scores)
        print("DeepEval rag:", rag_scores)

        results.append({"id": qid, "mode": "graph", **graph_scores})
        results.append({"id": qid, "mode": "rag", **rag_scores})

        # 4) latency
        lat_g = extract_latency_ms(resp_graph)
        lat_r = extract_latency_ms(resp_rag)
        if lat_g is not None:
            latency["graph"].append(lat_g)
        if lat_r is not None:
            latency["rag"].append(lat_r)

    # --- aggregate ---
    def avg(field: str, mode: str) -> Optional[float]:
        vals = [r[field] for r in results if r["mode"] == mode and field in r]
        return (sum(vals) / len(vals)) if vals else None

    fields = [f"p@{TOP_K}", f"r@{TOP_K}", f"mrr@{TOP_K}", f"ndcg@{TOP_K}"]

    print(f"\n=== RANKING METRICS (DeepEval) @K={TOP_K} ===")
    for mode in ["graph", "rag"]:
        row = {f: avg(f, mode) for f in fields}
        print(mode, row)

    print("\n=== LATENCY (ms) ===")
    for mode in ["graph", "rag"]:
        print(mode, summarize_latency(latency[mode]))


if __name__ == "__main__":
    run()
