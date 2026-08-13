import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.ai.ollama import ask_chat, resolve_model
from app.core.settings import ALLOWED_EXTENSIONS, TOOL_BY_EXTENSION
from app.tools import available_tools
from app.tools.run.naming import rename_file
from app.utils.language import detect_language, language_instruction

router = APIRouter(prefix='/chat', tags=['chat'])

RENAME_MENTION_PATTERN = re.compile(r'@tool_rename\b', re.IGNORECASE)
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


@router.post('')
async def chat(
    message: str = Form(...),
    model: str = Form(None),
    image: UploadFile = File(None),
    document: UploadFile = File(None),
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
    unknown_mentions = UNKNOWN_TOOL_PATTERN.findall(message)
    if rename_requested and document is None:
        raise HTTPException(
            status_code=400,
            detail='Tool mentions require an attached document',
        )
    if unknown_mentions:
        bad_tool = next(
            (m for m in unknown_mentions if m.lower() != 'rename'), None
        )
        if bad_tool is not None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown tool '@tool_{bad_tool}'",
            )

    extension = None
    if document is not None:
        extension = Path(document.filename or '').suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported document type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            )

    temp_path = None
    try:
        content = await image.read() if image is not None else None

        tools = None
        tool_hint = None
        if document is not None:
            data = await document.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as f:
                f.write(data)
                temp_path = f.name

        # Language is only enforced in tool flows (document/rename),
        # never in normal chat. Defaults to English when undetected.
        lang_instruction = None
        if rename_requested or document is not None:
            lang_instruction = language_instruction(detect_language(message))

        if rename_requested:
            tools = {'rename_file': rename_file}
            tool_hint = (
                f"{lang_instruction} "
                f"The user attached a document at '{temp_path}' and wants "
                "to rename it. You MUST call the tool 'rename_file' with the "
                "'file_path' argument set to that exact path. Do not reply "
                "without calling the tool. The client applies the rename with "
                "the name returned by the tool, so after calling it confirm to "
                "the user that the file has been renamed, e.g. 'The file was "
                "renamed to <name>'."
            )
            message = RENAME_MENTION_PATTERN.sub('', message).strip()
            if not message:
                message = 'Rename the attached document.'
        elif document is not None:
            tools = available_tools
            tool_name = TOOL_BY_EXTENSION[extension]
            tool_hint = (
                f"{lang_instruction} "
                f"The user attached a document at '{temp_path}'. "
                f"Use the tool '{tool_name}' to read it and answer the "
                "user's question. You cannot execute actions on the file "
                "(rename, move, delete, edit, etc.). If the user asks for "
                "an action you cannot perform, respond with exactly this "
                f"meaning: '{ACTION_NOT_SUPPORTED_MESSAGE}', briefly and "
                "without extra explanation. Never reveal, mention, or link "
                "the path of the attached document."
            )

        if rename_requested:
            response, tool_results = await run_in_threadpool(
                ask_chat, message, content, tools, tool_hint,
                model=effective_model,
                return_tool_results=True,
            )
            new_name = tool_results.get('rename_file')
            if not new_name:
                raise HTTPException(
                    status_code=502,
                    detail='The model did not generate a file name',
                )
            if new_name.startswith('Error'):
                raise HTTPException(status_code=502, detail=new_name)
            return {'response': _redact_path(response, temp_path), 'new_name': new_name}

        response = await run_in_threadpool(ask_chat, message, content, tools, tool_hint, model=effective_model)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f'Error querying the model: {e}')
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)

    return {'response': _redact_path(response, temp_path)}
