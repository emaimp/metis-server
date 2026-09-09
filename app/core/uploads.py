from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

from app.core.settings import MAX_UPLOAD_BYTES, MIME_BY_EXTENSION


def validate_upload(filename: str, data: bytes, allowed_extensions: set[str]) -> str:
    """Validate an uploaded file's extension and size; raise HTTPException 400 on failure."""
    ext = Path(filename or '').suffix.lower()
    if not ext or ext not in allowed_extensions:
        allowed = ', '.join(sorted(allowed_extensions))
        raise HTTPException(status_code=400, detail=f'Unsupported file type. Allowed: {allowed}')
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f'File exceeds the maximum size of {MAX_UPLOAD_BYTES} bytes',
        )
    return ext


def content_type_for(extension: str) -> str:
    """Return the MIME type for an extension, or a generic binary type."""
    return MIME_BY_EXTENSION.get(extension, 'application/octet-stream')



def build_attachment_hint(attachment_id: str, tool_name: str, filename: str) -> str:
    """Hint telling the model it must read an attached document (by id) with the given tool."""
    return (
        f"The user attached a document (id: '{attachment_id}', filename: '{filename}'). "
        f"Use the tool '{tool_name}' with attachment_id='{attachment_id}' to read it and "
        "answer the user's question. You cannot execute actions on the file "
        "(rename, move, delete, edit, etc.)."
    )