# src/api/main.py
from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.clients.llm_client import LLMClient
from src.database.database_service import DatabaseService
from src.services.ingestion_service import IngestionService

from src.agent.tools import AgentTools
from src.agent.agent_service import AgentService

from src.api.routes import router as agent_router
from src.api.health import router as health_router


def create_app() -> FastAPI:
    app = FastAPI(title="Foodify LLM Agent API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- singletons (projektowo OK) ----
    llm_client = LLMClient()
    db_service = DatabaseService()
    ingestion_service = IngestionService(db=db_service, llm=llm_client)
    tools = AgentTools(db=db_service, ingestion=ingestion_service)
    agent_service = AgentService(llm=llm_client, tools=tools)

    # ---- dependency injection (najprostsze możliwe) ----
    app.dependency_overrides[AgentService] = lambda: agent_service

    # ---- routes ----
    app.include_router(agent_router)
    app.include_router(health_router)

    return app


app = create_app()
