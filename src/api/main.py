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

from src.config.config import DB_URI, DB_USER, DB_PASSWORD, DB_DATABASE, RAW_DATASET_PATH



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
    db_service = DatabaseService(DB_URI, DB_USER, DB_PASSWORD, DB_DATABASE)
    ingestion_service = IngestionService(db_service, llm_client)
    tools = AgentTools(db=db_service, ingestion=ingestion_service)
    agent_service = AgentService(llm=llm_client, tools=tools)

    # ---- dependency injection ----
    # app.dependency_overrides[LLMClient] = lambda: llm_client
    # app.dependency_overrides[DatabaseService] = lambda: db_service
    # app.dependency_overrides[IngestionService] = lambda: ingestion_service
    # app.dependency_overrides[AgentTools] = lambda: tools
    # app.dependency_overrides[AgentService] = lambda: agent_service

    app.state.llm_client = llm_client
    app.state.db_service = db_service
    app.state.ingestion_service = ingestion_service
    app.state.tools = tools
    app.state.agent_service = agent_service

    # ---- routes ----
    app.include_router(agent_router)
    app.include_router(health_router)

    return app


app = create_app()
