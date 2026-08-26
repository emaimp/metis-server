import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.ai.ollama import ask_chat, resolve_model
from app.core.settings import DOCUMENT_EXTENSIONS, TOOL_BY_EXTENSION
from app.tools import available_tools
from app.tools.run.categorize import categorize_file
from app.tools.run.naming import rename_file

router = APIRouter(prefix='/chat', tags=['chat'])

RENAME_MENTION_PATTERN = re.compile(r'@tool_rename\b', re.IGNORECASE)
CATEGORIZE_MENTION_PATTERN = re.compile(r'@tool_categorize\b', re.IGNORECASE)
UNKNOWN_TOOL_PATTERN = re.compile(r'@tool_(\w+)', re.IGNORECASE)

ACTION_NOT_SUPPORTED_MESSAGE = (
    "I can't perform that action. I can only read and analyze "
    "the content of the attached document."
)


def _redact_path(response: str, temp_path: str | None) -> str:
    """Remove any occurrence of the temp file path from the response."""
    if temp_path:
        response = response.replace(temp_path, '[attached file]')
    return response


def _save_temp_bytes(data: bytes, extension: str) -> str:
    """Write raw bytes to a temporary file and return its path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as f:
        f.write(data)
        return f.name


async def _save_temp_document(document: UploadFile, extension: str) -> str:
    """Write the uploaded document to a temporary file and return its path."""
    data = await document.read()
    return _save_temp_bytes(data, extension)


async def _run_chat(
    message: str,
    content: bytes | None,
    tools: dict | None,
    tool_hint: str | None,
    model: str,
):
    """Run ask_chat in a thread pool."""
    return await run_in_threadpool(
        ask_chat, message, content, tools, tool_hint,
        model=model,
    )


def _require_tool_result(result: str, missing_detail: str) -> str:
    """Check a tool result string, raising 502 on missing or error."""
    if not result:
        raise HTTPException(status_code=502, detail=missing_detail)
    if result.startswith('Error'):
        raise HTTPException(status_code=502, detail=result)
    return result


def _document_hint(temp_path: str, tool_name: str) -> str:
    """Hint for answering questions about an attached file."""
    return (
        f"The user attached a file at '{temp_path}'. "
        f"Use the tool '{tool_name}' to read it and answer the "
        "user's question. You cannot execute actions on the file "
        "(rename, move, delete, edit, etc.). If the user asks for "
        "an action you cannot perform, respond with exactly this "
        f"meaning: '{ACTION_NOT_SUPPORTED_MESSAGE}', briefly and "
        "without extra explanation. Never reveal, mention, or link "
        "the path of the attached file."
    )


@router.post('')
async def chat(
    message: str = Form(...),
    model: str = Form(None),
    image: UploadFile = File(None),
    document: UploadFile = File(None),
    existing_categories: str = Form(''),
    existing_files: str = Form(''),
):
    """
    Free chat with the model. Image and document (.txt, .pdf, .docx) are optional:
    if attached, the model can answer questions about them. Tool mentions
    (@tool_x) in the message trigger the corresponding tool.

    The optional `model` form field overrides the server default (OLLAMA_MODEL
    in .env) for this request only.
    """
    try:
        effective_model = resolve_model(model)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    rename_requested = bool(RENAME_MENTION_PATTERN.search(message))
    categorize_requested = bool(CATEGORIZE_MENTION_PATTERN.search(message))
    unknown_mentions = UNKNOWN_TOOL_PATTERN.findall(message)
    if (rename_requested or categorize_requested) and document is None and image is None:
        raise HTTPException(
            status_code=400,
            detail='Tool mentions require an attached document or image',
        )
    if unknown_mentions:
        bad_tool = next(
            (
                m for m in unknown_mentions
                if m.lower() not in ('rename', 'categorize')
            ),
            None,
        )
        if bad_tool is not None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown tool '@tool_{bad_tool}'",
            )

    extension = None
    if document is not None:
        extension = Path(document.filename or '').suffix.lower()
        if extension not in DOCUMENT_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported document type. Allowed: {', '.join(sorted(DOCUMENT_EXTENSIONS))}",
            )

    temp_path = None
    try:
        content = await image.read() if image is not None else None

        if document is not None:
            temp_path = await _save_temp_document(document, extension)
        elif (rename_requested or categorize_requested) and content is not None:
            image_ext = Path(image.filename or '').suffix.lower()
            if image_ext not in TOOL_BY_EXTENSION:
                image_ext = '.png'
            temp_path = _save_temp_bytes(content, image_ext)
            content = None

        tools = None
        tool_hint = None

        if rename_requested:
            instruction = RENAME_MENTION_PATTERN.sub('', message).strip()
            files = (
                [f.strip() for f in existing_files.split(',') if f.strip()]
                if existing_files else None
            )
            result = await run_in_threadpool(
                rename_file, temp_path, files, instruction, effective_model,
            )
            new_name = _require_tool_result(
                result, 'The tool did not generate a file name',
            )
            return {'new_name': new_name}

        if categorize_requested:
            instruction = CATEGORIZE_MENTION_PATTERN.sub('', message).strip()
            categories = [
                c.strip() for c in existing_categories.split(',') if c.strip()
            ] if existing_categories else None
            result = await run_in_threadpool(
                categorize_file, temp_path, categories, instruction,
                effective_model,
            )
            category = _require_tool_result(
                result, 'The tool did not categorize the file',
            )
            return {'category': category}

        if document is not None:
            tools = available_tools
            tool_hint = _document_hint(temp_path, TOOL_BY_EXTENSION[extension])

        response = await _run_chat(message, content, tools, tool_hint, effective_model)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f'Error querying the model: {e}')
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)

    return {'response': _redact_path(response, temp_path)}
