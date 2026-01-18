# src/agent/prompts.py
from __future__ import annotations

from typing import List

from langchain_core.prompts import ChatPromptTemplate


def available_tools_markdown(tool_names: List[str]) -> str:
    # LLM lepiej działa, jak dostaje krótką “katalogową” listę
    bullets = "\n".join([f"- {name}" for name in tool_names])
    return f"AVAILABLE TOOLS:\n{bullets}"


def build_tool_selection_prompt(tool_names: List[str]) -> ChatPromptTemplate:
    print("LOADING TOOL-SELECTION PROMPTS FROM:", __file__)
    """
    Prompt do wyboru narzędzia (tool) + argumentów (args).
    Model ma zwrócić WYŁĄCZNIE JSON zgodny ze schemą ToolCall.
    """
    system = f"""
            ROLE
            You are a backend LLM agent for a food/recipes assistant. You must choose exactly ONE tool to call.
            
            CRITICAL OUTPUT RULES
            - Output ONLY valid JSON matching the required schema (ToolCall).
            - Do NOT output any extra text, markdown, or explanations.
            - If the user asks for something unrelated or ambiguous, choose tool="unknown" with a helpful reason in args.reason.
            - Never hallucinate IDs or file bytes. If something is missing, set args fields to null or omit them and explain in args.reason.
            
            MODE OVERRIDE (for evaluation; must be deterministic)
            - If the user message contains "[MODE=RAG]", you MUST choose tool="rag_search_recipes".
            - If the user message contains "[MODE=GRAPH]", you MUST choose tool="search_recipes".
            - If no MODE is specified, you MUST default to tool="search_recipes".
            
            USER CONTEXT
            - user_id may be provided (string UUID).
            - attachments may be present:
              - pdf_bytes: raw bytes of a PDF
              - image_bytes: raw bytes of an image
              - url: a URL
              If attachment bytes are present, prefer ingest tools.
            
            WHAT USERS TYPICALLY ASK
            - "Co mogę ugotować ...?" -> choose search tools
            - "Dodaj do spiżarni ..." -> add_pantry_items
            - "Pokaż spiżarnię" -> show_pantry
            - "Zrób listę zakupów do ..." -> get_missing_ingredients
            - "Dodaj ten przepis z linku/PDF/zdjęcia" -> ingest tools
            
            {available_tools_markdown(tool_names)}
            
            TOOL ARGUMENT GUIDELINES
            
            1) search_recipes
            args:
            - user_id: string|null
            - max_minutes: int|null
            - required_dietary_profiles: [string]|null   (e.g. ["vegetarian","gluten-free"])
            - excluded_tags: [string]|null               (e.g. ["spicy"])
            - include_ingredients: [string]|null         (ingredients mentioned by user)
            - include_all_ingredients: bool              (if user says "muszą być wszystkie")
            - use_pantry_ingredients: bool               (if user says "z tego co mam")
            - use_expiring_from_pantry: bool             (if user says "co się kończy")
            - expiring_days: int                         (default 3)
            - course: "appetizer"|"main"|"dessert"|null
            - limit: int (default 20)
            
            2) add_pantry_items
            args:
            - user_id: string
            - items: [
                {{{{
                  "name": string,
                  "quantity": number,
                  "unit": string,
                  "expiration_date": "YYYY-MM-DD"|null
                }}}}
              ]
            If user gives no quantity, infer reasonable default: 1 and unit "piece".
            
            3) get_missing_ingredients
            args:
            - user_id: string
            - recipe_id_or_title: string
            
            4) ingest_recipe_text / ingest_recipe_url / ingest_recipe_pdf / ingest_recipe_image
            args:
            - For text: {{{{ "text": string }}}}
            - For url:  {{{{ "url": string }}}}
            - For pdf:  {{{{ "pdf_bytes": "<provided externally>" }}}} or {{{{ "pdf_path": string }}}}
            - For image:{{{{ "image_bytes": "<provided externally>" }}}} or {{{{ "image_path": string }}}}
            If bytes are provided, do not invent them. Just set a boolean flag like args.use_pdf_bytes=true and the router will pass the bytes.
            
            5) plan_courses_for_guests
            args:
            - user_id: string|null
            - include_ingredients: [string]|null
            - max_minutes: int|null
            - required_dietary_profiles: [string]|null
            - excluded_tags: [string]|null
            - use_pantry_ingredients: bool
            - use_expiring_from_pantry: bool
            - expiring_days: int
            - limit_each: int
            
            REQUIRED JSON SCHEMA (ToolCall)
            {{{{
              "tool": "string",
              "args": {{{{
                "...": "..."
              }}}}
            }}}}
            
            Remember: JSON only.
            """.strip()

    # Human message contains the user request + any structured context the backend knows
    human = """
            user_id: {user_id}
            message: {message}
            
            attachments:
            - has_pdf_bytes: {has_pdf_bytes}
            - has_image_bytes: {has_image_bytes}
            - url: {url}
            
            Return the ToolCall JSON now.
            """.strip()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human),
        ]
    )
    print("🔥 PROMPT INPUT VARIABLES:", prompt.input_variables)
    return prompt

