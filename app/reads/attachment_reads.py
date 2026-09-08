from __future__ import annotations

from app.core.settings import MAX_CHARS
from app.reads.docx import extract_docx_from_bytes
from app.reads.pdf import extract_pdf_from_bytes
from app.reads.txt import extract_txt_from_bytes
from app.repositories.chats import get_attachment_db

MAX_PDF_PAGES = 30


def _truncate(content: str) -> str:
    if len(content) > MAX_CHARS:
        content = content[:MAX_CHARS] + (
            f"\n\n[... content truncated, only the first {MAX_CHARS} characters are shown ...]"
        )
    return content


def make_attachment_tools(chat_id: str) -> dict:
    """Build request-scoped read tools that load files from this chat's attachments in the database.

    Each tool takes the `attachment_id` given to the model in the hint and reads the raw
    bytes from the database (no temp files involved).
    """

    def _load(attachment_id: str) -> tuple[str, bytes] | str:
        attachment = get_attachment_db(chat_id, attachment_id)
        if attachment is None:
            return f"Error: The attachment '{attachment_id}' does not exist."
        return attachment["filename"], attachment["data"]

    def read_txt(attachment_id: str) -> str:
        """Reads a text document (.txt) stored as a chat attachment.

        Args:
            attachment_id: The id of the attached text document.
        """
        loaded = _load(attachment_id)
        if isinstance(loaded, str):
            return loaded
        _filename, data = loaded
        return extract_txt_from_bytes(data)

    def read_pdf(attachment_id: str) -> str:
        """Reads a PDF document stored as a chat attachment.

        Args:
            attachment_id: The id of the attached PDF document.
        """
        loaded = _load(attachment_id)
        if isinstance(loaded, str):
            return loaded
        _filename, data = loaded
        content = extract_pdf_from_bytes(data, MAX_PDF_PAGES)
        if content.startswith("Error"):
            return content
        return _truncate(content)

    def read_docx(attachment_id: str) -> str:
        """Reads a Word document (.docx) stored as a chat attachment.

        Args:
            attachment_id: The id of the attached Word document.
        """
        loaded = _load(attachment_id)
        if isinstance(loaded, str):
            return loaded
        _filename, data = loaded
        content = extract_docx_from_bytes(data)
        if content.startswith("Error"):
            return content
        return _truncate(content)

    return {"read_txt": read_txt, "read_pdf": read_pdf, "read_docx": read_docx}