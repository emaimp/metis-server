from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, File, Form, HTTPException, Query, Response, UploadFile
from starlette.concurrency import run_in_threadpool

from app.ai.ollama import ask_chat, resolve_model
from app.core.markdown import strip_markdown
from app.core.settings import DOCUMENT_EXTENSIONS, IMAGE_EXTENSIONS, TOOL_BY_EXTENSION
from app.core.time_utils import now_iso
from app.core.uploads import (
    build_attachment_hint,
    content_type_for,
    validate_upload,
)
from app.repositories.chats import (
    add_audio_db,
    add_messages_db,
    chat_exists_db,
    create_attachment_db,
    create_chat_db,
    delete_attachments_db,
    delete_audio_by_message_db,
    delete_chat_db,
    get_attachment_db,
    get_audio_by_message_db,
    get_audio_db,
    get_chat_db,
    get_message_db,
    link_attachments_to_message_db,
    list_chats_db,
    update_title_db,
)
from app.schemas.chats import (
    AudioResponse,
    ChatDetailResponse,
    ChatListItem,
    ChatSummary,
    CreateChatRequest,
    UpdateTitleRequest,
)
from app.reads.attachment_reads import make_attachment_tools
from app.tools.run.categorize import categorize_file
from app.tools.run.naming import rename_file

RENAME_MENTION_PATTERN = re.compile(r"@tool_rename\b", re.IGNORECASE)
CATEGORIZE_MENTION_PATTERN = re.compile(r"@tool_categorize\b", re.IGNORECASE)
UNKNOWN_TOOL_PATTERN = re.compile(r"@tool_(\w+)", re.IGNORECASE)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chats", tags=["chats"])


class NoSpeakableTextError(Exception):
    """Raised when the TTS input has no speakable text after markdown stripping."""


def _require_tool_result(result: str, missing_detail: str) -> str:
    """Check an action tool result string, raising 502 on missing or error."""
    if not result:
        raise HTTPException(status_code=502, detail=missing_detail)
    if result.startswith("Error"):
        raise HTTPException(status_code=502, detail=result)
    return result


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


async def _synthesize_and_store(chat_id: str, message_id: str, content: str, voice: str | None) -> dict:
    """Run the TTS pipeline for a message, persist the WAV as BLOB, and return meta + base64."""
    try:
        def _tts_sync():
            # Lazy imports: omnivoice pulls in torch (heavy), only needed for TTS.
            from app.ai.omnivoice import SAMPLE_RATE, synthesize
            from app.ai.tts.encoding import ndarray_to_wav_bytes

            tts_text = strip_markdown(content)
            if not tts_text or not tts_text.strip():
                raise NoSpeakableTextError()

            audio, sample_rate = synthesize(tts_text, instruct=voice)
            return ndarray_to_wav_bytes(audio, sample_rate), sample_rate

        wav_bytes, wav_sample_rate = await run_in_threadpool(_tts_sync)
        audio_meta = await run_in_threadpool(add_audio_db, chat_id, message_id, wav_bytes, wav_sample_rate)
        audio_meta = {k: audio_meta[k] for k in ("id", "mime_type", "sample_rate", "size", "created_at")}
        from app.ai.tts.encoding import wav_bytes_to_base64

        audio_base64 = wav_bytes_to_base64(wav_bytes)
        return {
            "audio": audio_meta,
            "audio_base64": audio_base64,
            "mime_type": audio_meta["mime_type"],
            "sample_rate": wav_sample_rate,
        }
    except NoSpeakableTextError:
        raise HTTPException(
            status_code=502,
            detail="Model response contains no speakable text after markdown stripping",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("TTS generation failed")
        raise HTTPException(status_code=502, detail=f"TTS generation failed: {e}")


@router.post("/{chat_id}/messages")
async def create_message(
    chat_id: str,
    message: str = Form(...),
    model: str | None = Form(None),
    image: UploadFile | None = File(None),
    document: UploadFile | None = File(None),
    existing_files: str = Form(""),
    existing_categories: str = Form(""),
    tts: bool = Form(False),
    voice: str | None = Form(None),
):
    """Send a single-turn message (optionally with an image, document, tool mentions and/or TTS) and persist the exchange.

    - `@tool_rename` / `@tool_categorize` mentions (require an attached file) run the
      action tools and return `{"new_name" | "category", ...}`; the exchange is persisted.
    - Without mentions, the message goes to the model (image bytes and/or the document
      read tools) and the model response is returned.
    - With `tts=true`, the assistant response is synthesized, stored and returned inline.
    """
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

    rename_requested = bool(RENAME_MENTION_PATTERN.search(message))
    categorize_requested = bool(CATEGORIZE_MENTION_PATTERN.search(message))
    unknown_mentions = UNKNOWN_TOOL_PATTERN.findall(message)

    # Read and validate optional uploads, and persist them BEFORE calling the model
    # so the read tools can load them from the database (no temp paths involved).
    image_data: bytes | None = None
    document_data: bytes | None = None
    image_meta: dict | None = None
    document_meta: dict | None = None
    image_ext: str | None = None
    document_ext: str | None = None
    if image is not None:
        image_data = await image.read()
        image_ext = validate_upload(image.filename, image_data, IMAGE_EXTENSIONS)
        image_meta = await run_in_threadpool(
            create_attachment_db, chat_id, image.filename, content_type_for(image_ext), image_data,
        )
    if document is not None:
        document_data = await document.read()
        document_ext = validate_upload(document.filename, document_data, DOCUMENT_EXTENSIONS)
        document_meta = await run_in_threadpool(
            create_attachment_db, chat_id, document.filename, content_type_for(document_ext), document_data,
        )
    pending_attachment_ids = [m["id"] for m in (image_meta, document_meta) if m is not None]

    async def _discard_pending_attachments():
        if pending_attachment_ids:
            await run_in_threadpool(delete_attachments_db, pending_attachment_ids)

    # Validate tool mentions
    if unknown_mentions:
        bad_tool = next(
            (m for m in unknown_mentions if m.lower() not in ("rename", "categorize")),
            None,
        )
        if bad_tool is not None:
            await _discard_pending_attachments()
            raise HTTPException(status_code=400, detail=f"Unknown tool '@tool_{bad_tool}'")
    if (rename_requested or categorize_requested) and not pending_attachment_ids:
        raise HTTPException(status_code=400, detail="Tool mentions require an attached document or image")

    user_created_at = now_iso()
    # Action tools: @tool_rename / @tool_categorize (require an attachment, persisted above)
    if rename_requested or categorize_requested:
        target_data = document_data if document_data is not None else image_data
        target_ext = document_ext if document_ext is not None else image_ext
        t0 = time.perf_counter()
        try:
            if rename_requested:
                instruction = RENAME_MENTION_PATTERN.sub("", message).strip()
                files = (
                    [f.strip() for f in existing_files.split(",") if f.strip()]
                    if existing_files
                    else None
                )
                result = await run_in_threadpool(
                    rename_file, target_data, target_ext, files, instruction, effective_model,
                )
                result = _require_tool_result(result, "The tool did not generate a file name")
            else:
                instruction = CATEGORIZE_MENTION_PATTERN.sub("", message).strip()
                categories = (
                    [c.strip() for c in existing_categories.split(",") if c.strip()]
                    if existing_categories
                    else None
                )
                result = await run_in_threadpool(
                    categorize_file, target_data, target_ext, categories, instruction, effective_model,
                )
                result = _require_tool_result(result, "The tool did not categorize the file")
        except HTTPException:
            await _discard_pending_attachments()
            raise
        except Exception as e:
            logger.exception("Tool execution failed")
            await _discard_pending_attachments()
            raise HTTPException(status_code=502, detail=f"Error running the tool: {e}")
        t1 = time.perf_counter()
        response_time_ms = int((t1 - t0) * 1000)

        # Persist the exchange: user message (as typed) + assistant message (tool result)
        try:
            user_msg, assistant_msg = await run_in_threadpool(
                add_messages_db, chat_id, message, result, effective_model,
                response_time_ms, user_created_at, now_iso(),
            )
        except ValueError as e:
            await _discard_pending_attachments()
            raise HTTPException(status_code=404, detail=str(e))
        await run_in_threadpool(link_attachments_to_message_db, pending_attachment_ids, user_msg["id"])
        user_msg["attachments"] = [
            {k: m[k] for k in ("id", "filename", "content_type", "size")}
            for m in (image_meta, document_meta) if m is not None
        ]
        updated_chat = await run_in_threadpool(get_chat_db, chat_id)
        payload = {"new_name": result} if rename_requested else {"category": result}
        payload.update({
            "user_message": user_msg,
            "assistant_message": assistant_msg,
            "chat": updated_chat,
        })
        return payload

    # Normal flow: ask the model (image bytes and/or document read tools from the database)
    t0 = time.perf_counter()
    try:
        if document_meta is not None:
            doc_tools = make_attachment_tools(chat_id)
            tool_name = TOOL_BY_EXTENSION[document_ext]
            response = await run_in_threadpool(
                ask_chat, message,
                image=image_data,
                available_tools=doc_tools,
                tool_hint=build_attachment_hint(document_meta["id"], tool_name, document_meta["filename"]),
                model=effective_model,
            )
        else:
            response = await run_in_threadpool(
                ask_chat, message, image=image_data, model=effective_model,
            )
    except Exception as e:
        logger.exception("ask_chat failed")
        await _discard_pending_attachments()
        raise HTTPException(status_code=502, detail=f"Error querying the model: {e}")
    t1 = time.perf_counter()
    response_time_ms = int((t1 - t0) * 1000)
    assistant_created_at = now_iso()

    if not response or not response.strip():
        await _discard_pending_attachments()
        raise HTTPException(status_code=502, detail="Empty model response")

    # Persist messages and link the persisted attachments to the user message
    try:
        user_msg, assistant_msg = await run_in_threadpool(
            add_messages_db, chat_id, message, response, effective_model,
            response_time_ms, user_created_at, assistant_created_at,
        )
    except ValueError as e:
        await _discard_pending_attachments()
        raise HTTPException(status_code=404, detail=str(e))
    if pending_attachment_ids:
        await run_in_threadpool(link_attachments_to_message_db, pending_attachment_ids, user_msg["id"])
        user_msg["attachments"] = [
            {k: m[k] for k in ("id", "filename", "content_type", "size")}
            for m in (image_meta, document_meta) if m is not None
        ]

    # Fetch updated chat detail
    updated_chat = await run_in_threadpool(get_chat_db, chat_id)

    # Optional TTS: synthesize the assistant response, store the WAV and return it inline.
    # If TTS fails, the messages remain persisted and the audio can be regenerated on demand.
    audio = None
    user_msg["audio"] = None
    assistant_msg["audio"] = None
    if tts:
        audio = await _synthesize_and_store(chat_id, assistant_msg["id"], assistant_msg["content"], voice)
        assistant_msg["audio"] = audio["audio"]

    return {
        "user_message": user_msg,
        "assistant_message": assistant_msg,
        "chat": updated_chat,
        "audio_base64": audio["audio_base64"] if audio else None,
        "mime_type": audio["mime_type"] if audio else None,
        "sample_rate": audio["sample_rate"] if audio else None,
    }


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


@router.get("/{chat_id}/audios/{audio_id}")
async def get_audio(chat_id: str, audio_id: str):
    """Download a stored TTS audio (WAV) from a chat. Returns 404 if not found."""
    audio = await run_in_threadpool(get_audio_db, chat_id, audio_id)
    if not audio:
        raise HTTPException(status_code=404, detail="Audio not found")
    return Response(content=audio["data"], media_type=audio["mime_type"])


@router.post("/{chat_id}/messages/{message_id}/audio", response_model=AudioResponse)
async def create_message_audio(chat_id: str, message_id: str, voice: str | None = Query(None)):
    """Generate (or reuse) the TTS audio for an assistant message, on demand.

    If the message already has a stored audio it is returned as-is (idempotent).
    The `voice` query param only applies when the audio does not exist yet.
    """
    if not await run_in_threadpool(chat_exists_db, chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    message = await run_in_threadpool(get_message_db, chat_id, message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    if message["role"] != "assistant":
        raise HTTPException(status_code=400, detail="Audio is only available for assistant messages")

    # Idempotent: reuse the stored audio if present (no regeneration)
    existing = await run_in_threadpool(get_audio_by_message_db, chat_id, message_id)
    if existing:
        from app.ai.tts.encoding import wav_bytes_to_base64

        meta = {key: existing[key] for key in ("id", "mime_type", "sample_rate", "size", "created_at")}
        return {
            "audio": meta,
            "audio_base64": wav_bytes_to_base64(existing["data"]),
            "mime_type": existing["mime_type"],
            "sample_rate": existing["sample_rate"],
        }

    return await _synthesize_and_store(chat_id, message_id, message["content"], voice)


@router.delete("/{chat_id}/messages/{message_id}/audio")
async def delete_message_audio(chat_id: str, message_id: str):
    """Delete the stored TTS audio for an assistant message, so it can be regenerated.

    After deletion, `GET /chats/{chat_id}` shows `audio: null` for the message
    and `POST .../audio` synthesizes a fresh audio on demand.
    """
    if not await run_in_threadpool(chat_exists_db, chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    message = await run_in_threadpool(get_message_db, chat_id, message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    if message["role"] != "assistant":
        raise HTTPException(status_code=400, detail="Audio is only available for assistant messages")
    deleted = await run_in_threadpool(delete_audio_by_message_db, chat_id, message_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Audio not found")
    return {"detail": "Audio deleted"}
