# src/agent/agent_service.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional
from unittest.mock import call

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from src.clients.llm_client import LLMClient
from src.agent.prompts import build_tool_selection_prompt
from src.agent.tools import AgentTools, ToolName, ToolResult


logger = logging.getLogger(__name__)


class ToolCall(BaseModel):
    tool: str = Field(..., description="Name of the tool to call")
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool")


@dataclass
class AgentMessage:
    user_id: Optional[str]
    message: str
    # Optional structured context (API layer can fill these)
    pdf_bytes: Optional[bytes] = None
    image_bytes: Optional[bytes] = None
    url: Optional[str] = None


class AgentService:
    """
    Router: user message -> LLM tool selection -> call AgentTools -> ToolResult.
    Keeps API endpoints tiny.
    """

    def __init__(self, llm: LLMClient, tools: AgentTools):
        self._llm = llm
        self._tools = tools

        # Available tool names = enum values (strings)
        self._tool_names = [t.value for t in ToolName]
        self._prompt: ChatPromptTemplate = build_tool_selection_prompt(self._tool_names)

        # Structured output parser (LangChain)
        # NOTE: LLMClient uses AzureChatOpenAI under self._client.
        self._structured_llm = self._llm._client.with_structured_output(
            ToolCall,
            method="json_mode",
        )

    def handle_message(self, req: AgentMessage) -> ToolResult:
        """
        Main entry point called by API.

        Returns ToolResult with:
        - type: frontend rendering hint (e.g. recipes_list)
        - message: short text to display
        - data: payload
        """
        # 0) If attachment bytes exist, hard-route ingest (avoid LLM confusion)
        if req.pdf_bytes is not None:
            return self._tools.ingest_recipe_pdf(pdf_bytes=req.pdf_bytes)
        if req.image_bytes is not None:
            return self._tools.ingest_recipe_image(image_bytes=req.image_bytes)

        # 1) Ask LLM for tool selection (structured JSON)
        tool_call = self._select_tool(req)

        # 2) Execute tool
        return self._execute_tool(tool_call, req)

    def _select_tool(self, req: AgentMessage) -> ToolCall:
        # Simple URL hint: if API already extracted url, pass it; else LLM can detect from message
        has_pdf_bytes = req.pdf_bytes is not None
        has_image_bytes = req.image_bytes is not None

        try:
            pipeline = self._prompt | self._structured_llm
            parsed: ToolCall = pipeline.invoke(
                {
                    "user_id": req.user_id,
                    "message": req.message,
                    "has_pdf_bytes": has_pdf_bytes,
                    "has_image_bytes": has_image_bytes,
                    "url": req.url,
                }
            )
            # normalize
            parsed.tool = (parsed.tool or "").strip()
            if not isinstance(parsed.args, dict):
                parsed.args = {}
            return parsed
        except Exception as e:
            logger.exception("Tool selection failed: %s", e)
            return ToolCall(
                tool="unknown",
                args={"reason": f"Tool selection failed: {type(e).__name__}: {e}"},
            )

    def _execute_tool(self, call: ToolCall, req: AgentMessage) -> ToolResult:
        tool = (call.tool or "").strip()

        # Unknown / fallback
        if tool in ("unknown", "", None):
            reason = call.args.get("reason") if isinstance(call.args, dict) else None
            return ToolResult(
                type="answer",
                message=reason or "Nie jestem pewna co zrobić — doprecyzuj proszę (np. czas, składniki, dieta).",
                data={"tool": "unknown", "args": call.args},
            )

        # Validate tool name
        if tool not in self._tool_names:
            return ToolResult(
                type="error",
                message=f"Nieobsługiwane narzędzie: {tool}",
                data={"tool": tool, "args": call.args},
            )

        # Dispatch
        try:
            args = call.args or {}

            # Provide defaults / safe wiring:
            if tool == ToolName.INGEST_RECIPE_URL.value:
                # If args.url missing, try fallback from req.url
                url = args.get("url") or req.url
                if not url:
                    return ToolResult(
                        type="error",
                        message="Brakuje URL do ingestu.",
                        data={"tool": tool, "args": args},
                    )
                return self._tools.ingest_recipe_url(url=url)

            if tool == ToolName.INGEST_RECIPE_TEXT.value:
                text = args.get("text") or req.message
                return self._tools.ingest_recipe_text(text=text)

            if tool == ToolName.INGEST_RECIPE_PDF.value:
                # If model requested bytes, we should have them from API
                if req.pdf_bytes is not None:
                    return self._tools.ingest_recipe_pdf(pdf_bytes=req.pdf_bytes)
                return self._tools.ingest_recipe_pdf(pdf_path=args.get("pdf_path"))

            if tool == ToolName.INGEST_RECIPE_IMAGE.value:
                if req.image_bytes is not None:
                    return self._tools.ingest_recipe_image(image_bytes=req.image_bytes)
                return self._tools.ingest_recipe_image(image_path=args.get("image_path"))

            if tool == ToolName.CREATE_USER.value:
                return self._tools.create_user(
                    name=args.get("name"),
                    email=args.get("email"),
                    dietary_profiles=args.get("dietary_profiles"),
                )

            if tool == ToolName.SHOW_PANTRY.value:
                user_id = args.get("user_id") or req.user_id
                if not user_id:
                    return ToolResult(
                        type="error",
                        message="Brakuje user_id (nie wiem czyją spiżarnię pokazać).",
                        data={"tool": tool, "args": args},
                    )
                return self._tools.show_pantry(user_id=user_id)

            if tool == ToolName.ADD_PANTRY_ITEMS.value:
                user_id = args.get("user_id") or req.user_id
                if not user_id:
                    return ToolResult(
                        type="error",
                        message="Brakuje user_id (nie wiem komu dodać produkty).",
                        data={"tool": tool, "args": args},
                    )
                items = args.get("items") or []
                return self._tools.add_pantry_items(user_id=user_id, items=items)

            if tool in (
                ToolName.SEARCH_RECIPES.value,
                ToolName.SEARCH_FROM_PANTRY.value,
                ToolName.SEARCH_FROM_LIST.value,
                ToolName.SEARCH_EXPIRING.value,
                ToolName.SEARCH_UNDER_TIME.value,
                ToolName.PLAN_COURSES.value,
                ToolName.MISSING_INGREDIENTS.value,
                ToolName.RAG_SEARCH.value,
                ToolName.SEASONAL_RECIPES.value,
            ):
                return self._execute_search_like(tool, args, req)

            # Should never happen due to validation above, but safe:
            return ToolResult(
                type="error",
                message=f"Brak dispatcha dla narzędzia: {tool}",
                data={"tool": tool, "args": args},
            )

        except Exception as e:
            logger.exception("Tool execution failed (%s): %s", tool, e)
            return ToolResult(
                type="error",
                message=f"Błąd podczas wykonywania akcji ({tool}): {type(e).__name__}: {e}",
                data={"tool": tool, "args": call.args},
            )

    def _execute_search_like(self, tool: str, args: Dict[str, Any], req: AgentMessage) -> ToolResult:
        # Common user_id fallback
        if "user_id" not in args and req.user_id:
            args["user_id"] = req.user_id

        if tool == ToolName.SEARCH_RECIPES.value:
            return self._tools.search_recipes(**args)

        if tool == ToolName.SEARCH_FROM_PANTRY.value:
            user_id = args.get("user_id") or req.user_id
            if not user_id:
                return ToolResult(
                    type="error",
                    message="Brakuje user_id (nie wiem czyje pantry użyć).",
                    data={"tool": tool, "args": args},
                )
            args.pop("include_ingredients", None)  # from_pantry ignores include list
            return self._tools.search_recipes_from_pantry(**args)

        if tool == ToolName.SEARCH_FROM_LIST.value:
            ingredients = args.get("ingredients") or args.get("include_ingredients") or []
            return self._tools.search_recipes_from_list(ingredients=ingredients, **{k: v for k, v in args.items() if k not in ("ingredients",)})

        if tool == ToolName.SEARCH_EXPIRING.value:
            user_id = args.get("user_id") or req.user_id
            if not user_id:
                return ToolResult(
                    type="error",
                    message="Brakuje user_id (nie wiem czyje produkty sprawdzić).",
                    data={"tool": tool, "args": args},
                )

            days_raw = args.get("days", None)
            exp_raw = args.get("expiring_days", None)

            days = int(days_raw or exp_raw or 3)
            logger.warning("EXPIRING args BEFORE CLEAN: %s", call.args)
            logger.warning("EXPIRING args AFTER CLEAN: %s", args)
            logger.warning("EXPIRING days=%s user_id=%s", days, user_id)
            # We accept BOTH keys from LLM: "days" and "expiring_days"
            # 🔥 CRITICAL: delete BOTH keys BEFORE **args
            args = dict(args)  # make a shallow copy so we're sure we mutate the one we pass
            args.pop("days", None)
            args.pop("expiring_days", None)

            return self._tools.search_recipes_expiring(days=days, **args)
        # if tool == ToolName.SEARCH_EXPIRING.value:
        #     user_id = args.get("user_id") or req.user_id
        #     if not user_id:
        #         return ToolResult(
        #             type="error",
        #             message="Brakuje user_id (nie wiem czyje produkty sprawdzić).",
        #             data={"tool": tool, "args": args},
        #         )
        #     days = int(args.get("days") or args.get("expiring_days") or 3)
        #     # clean args
        #     # args.pop("days", None)
        #     return self._tools.search_recipes_expiring(days=days, **args)

        if tool == ToolName.SEARCH_UNDER_TIME.value:
            minutes = int(args.get("minutes") or args.get("max_minutes") or 30)
            limit = int(args.get("limit") or 20)
            return self._tools.search_recipes_under_time(minutes=minutes, limit=limit)

        if tool == ToolName.MISSING_INGREDIENTS.value:
            user_id = args.get("user_id") or req.user_id
            if not user_id:
                return ToolResult(
                    type="error",
                    message="Brakuje user_id (nie wiem czyją spiżarnię porównać).",
                    data={"tool": tool, "args": args},
                )
            rid = args.get("recipe_id_or_title") or args.get("recipe") or args.get("title")
            if not rid:
                return ToolResult(
                    type="error",
                    message="Brakuje nazwy/ID przepisu do listy zakupów.",
                    data={"tool": tool, "args": args},
                )
            return self._tools.get_missing_ingredients(user_id=user_id, recipe_id_or_title=rid)

        if tool == ToolName.RAG_SEARCH.value:
            query_text = args.get("query_text") or args.get("query") or req.message
            k = int(args.get("k") or 5)
            return self._tools.rag_search(query_text=query_text, k=k)

        if tool == ToolName.RAG_SEARCH_RECIPES.value:
            query_text = args.get("query_text") or args.get("query") or req.message

            k_chunks = int(args.get("k_chunks") or 30)
            limit_recipes = int(args.get("limit_recipes") or 5)
            chunks_per_recipe = int(args.get("chunks_per_recipe") or 3)

            max_minutes = args.get("max_minutes")
            max_minutes = int(max_minutes) if max_minutes is not None else None

            return self._tools.rag_search_recipes(
                query_text=query_text,
                k_chunks=k_chunks,
                limit_recipes=limit_recipes,
                chunks_per_recipe=chunks_per_recipe,
                max_minutes=max_minutes,
            )

        if tool == ToolName.PLAN_COURSES.value:
            return self._tools.plan_courses_for_guests(**args)

        if tool == ToolName.SEASONAL_RECIPES.value:
            season = args.get("season")  # może być None
            max_minutes = args.get("max_minutes")
            max_minutes = int(max_minutes) if max_minutes is not None else None
            limit = int(args.get("limit") or 20)

            return self._tools.seasonal_recipes(
                season=season,
                max_minutes=max_minutes,
                required_dietary_profiles=args.get("required_dietary_profiles"),
                excluded_tags=args.get("excluded_tags"),
                limit=limit,
            )