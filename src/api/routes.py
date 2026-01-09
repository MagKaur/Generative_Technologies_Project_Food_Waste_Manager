# src/api/routes/agent.py
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile

from src.agent.agent_service import AgentService, AgentMessage


router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/message")
async def agent_message(
    agent_service: AgentService,   # wstrzykiwany z main.py
    user_id: Optional[str] = Form(default=None),
    message: str = Form(...),
    url: Optional[str] = Form(default=None),
    pdf: Optional[UploadFile] = File(default=None),
    image: Optional[UploadFile] = File(default=None),
):
    pdf_bytes = await pdf.read() if pdf else None
    image_bytes = await image.read() if image else None

    req = AgentMessage(
        user_id=user_id,
        message=message,
        pdf_bytes=pdf_bytes,
        image_bytes=image_bytes,
        url=url,
    )

    result = agent_service.handle_message(req)
    return {
        "type": result.type,
        "message": result.message,
        "data": result.data,
    }
