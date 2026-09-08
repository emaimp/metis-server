from __future__ import annotations

import logging
import time
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from starlette.concurrency import run_in_threadpool

from app.ai.ollama import ask_chat, resolve_model
from app.core.settings import DOCUMENT_EXTENSIONS, IMAGE_EXTENSIONS, TOOL_BY_EXTENSION
from app.core.time_utils import now_iso
from app.core.uploads import (
    build_document_hint,
    content_type_for,
    save_temp_bytes,
    validate_upload,
)
from app.repositories.chats import (
    add_messages_db,
    chat_exists_db,
    create_chat_db,
    delete_chat_db,
    get_attachment_db,
    get_chat_db,
    list_chats_db,
    update_title_db,
)
from app.schemas.chats import (
    ChatDetailResponse,
    ChatListItem,
    ChatSummary,
    CreateChatRequest,
    CreateMessageResponse,
    UpdateTitleRequest,
)
from app.tools import available_tools

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
async def create_message(
    chat_id: str,
    message: str = Form(...),
    model: str | None = Form(None),
    image: UploadFile | None = File(None),
    document: UploadFile | None = File(None),
):
    """Send a single-turn message (optionally with an image or document) and persist the exchange."""
    message = message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message must be non-empty")
    # Fail fast before spending a model call on a missing chat
    if not await run_in_threadpool(chat_exists_db, chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")

    # Resolve model per message or use default
    try:
        effective_model = resolve_model(model)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Read and validate optional uploads before calling the model
    image_data: bytes | None = None
    document_data: bytes | None = None
    image_ext: str | None = None
    document_ext: str | None = None
    if image is not None:
        image_data = await image.read()
        image_ext = validate_upload(image.filename, image_data, IMAGE_EXTENSIONS)
    if document is not None:
        document_data = await document.read()
        document_ext = validate_upload(document.filename, document_data, DOCUMENT_EXTENSIONS)

    user_created_at = now_iso()
    t0 = time.perf_counter()
    temp_path = None
    try:
        if document_ext is not None:
            temp_path = save_temp_bytes(document_data, document_ext)
            tool_name = TOOL_BY_EXTENSION[document_ext]
            # Run in threadpool since ask_chat is blocking
            response = await run_in_threadpool(
                ask_chat, message,
                image=image_data,
                available_tools=available_tools,
                tool_hint=build_document_hint(temp_path, tool_name),
                model=effective_model,
            )
        else:
            response = await run_in_threadpool(
                ask_chat, message, image=image_data, model=effective_model,
            )
    except Exception as e:
        logger.exception("ask_chat failed")
        raise HTTPException(status_code=502, detail=f"Error querying the model: {e}")
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)
    t1 = time.perf_counter()
    response_time_ms = int((t1 - t0) * 1000)
    assistant_created_at = now_iso()

    if not response or not response.strip():
        raise HTTPException(status_code=502, detail="Empty model response")

    # Store messages and attachments, update chat title/updated_at
    attachments = []
    if image_data is not None:
        attachments.append({
            "filename": image.filename,
            "content_type": content_type_for(image_ext),
            "data": image_data,
        })
    if document_data is not None:
        attachments.append({
            "filename": document.filename,
            "content_type": content_type_for(document_ext),
            "data": document_data,
        })
    try:
        user_msg, assistant_msg = await run_in_threadpool(
            add_messages_db, chat_id, message, response, effective_model,
            response_time_ms, user_created_at, assistant_created_at, attachments,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Fetch updated chat detail
    updated_chat = await run_in_threadpool(get_chat_db, chat_id)
    return {"user_message": user_msg, "assistant_message": assistant_msg, "chat": updated_chat}


@router.get("/{chat_id}/attachments/{attachment_id}")
async def get_attachment(chat_id: str, attachment_id: str):
    """Download an uploaded file (image or document) from a chat. Returns 404 if not found."""
    attachment = await run_in_threadpool(get_attachment_db, chat_id, attachment_id)
    if not attachment:
        raise HTTPException(status_code=404, detail="Attachment not found")
    filename = attachment["filename"]
    headers = {
        "Content-Disposition": f"inline; filename*=UTF-8''{quote(filename)}",
    }
    return Response(
        content=attachment["data"],
        media_type=attachment["content_type"],
        headers=headers,
    )
