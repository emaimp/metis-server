from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from app.ai.ollama import ask_chat, resolve_model
from app.core.time_utils import now_iso
from app.repositories.chats import (
    add_messages_db,
    chat_exists_db,
    create_chat_db,
    delete_chat_db,
    get_chat_db,
    list_chats_db,
    update_title_db,
)
from app.schemas.chats import (
    ChatDetailResponse,
    ChatListItem,
    ChatSummary,
    CreateChatRequest,
    CreateMessageRequest,
    CreateMessageResponse,
    UpdateTitleRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chats", tags=["chats"])


@router.post("", response_model=ChatSummary)
async def create_chat(payload: CreateChatRequest):
    """Create a new chat and return its summary. The model is chosen per message."""
    chat = await run_in_threadpool(create_chat_db, payload.title)
    return chat


@router.get("", response_model=list[ChatListItem])
async def list_chats():
    """List all chats, newest first, with message counts."""
    chats = await run_in_threadpool(list_chats_db)
    return chats


@router.get("/{chat_id}", response_model=ChatDetailResponse)
async def get_chat(chat_id: str):
    """Get a chat with its full message history, or 404."""
    chat = await run_in_threadpool(get_chat_db, chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@router.patch("/{chat_id}", response_model=ChatSummary)
async def update_chat_title(chat_id: str, payload: UpdateTitleRequest):
    """Rename a chat. The repository caps the title at 200 chars."""
    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title must be non-empty")
    updated = await run_in_threadpool(update_title_db, chat_id, title)
    if not updated:
        raise HTTPException(status_code=404, detail="Chat not found")
    return updated


@router.delete("/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete a chat, or 404 if it does not exist."""
    deleted = await run_in_threadpool(delete_chat_db, chat_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"detail": "Chat deleted"}


@router.post("/{chat_id}/messages", response_model=CreateMessageResponse)
async def create_message(chat_id: str, payload: CreateMessageRequest):
    """Send a single-turn message and persist the user/assistant exchange."""
    message = payload.message
    if not message or not message.strip():
        raise HTTPException(status_code=400, detail="Message must be non-empty")
    # Fail fast before spending a model call on a missing chat
    if not await run_in_threadpool(chat_exists_db, chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")

    # Resolve model per message or use default
    try:
        effective_model = resolve_model(payload.model)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    user_created_at = now_iso()
    t0 = time.perf_counter()
    try:
        # Single-turn: call ask_chat with only current message (no history)
        # Run in threadpool since ask_chat is blocking
        response = await run_in_threadpool(ask_chat, message, model=effective_model)
    except Exception as e:
        logger.exception("ask_chat failed")
        raise HTTPException(status_code=502, detail=f"Error querying the model: {e}")
    t1 = time.perf_counter()
    response_time_ms = int((t1 - t0) * 1000)
    assistant_created_at = now_iso()

    if not response or not response.strip():
        raise HTTPException(status_code=502, detail="Empty model response")

    # Store messages and update chat title/updated_at
    try:
        user_msg, assistant_msg = await run_in_threadpool(
            add_messages_db, chat_id, message.strip(), response, effective_model, response_time_ms, user_created_at, assistant_created_at
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Fetch updated chat detail
    updated_chat = await run_in_threadpool(get_chat_db, chat_id)
    return {"user_message": user_msg, "assistant_message": assistant_msg, "chat": updated_chat}
