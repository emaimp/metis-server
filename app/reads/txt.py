from app.core.settings import MAX_CHARS


# Text extraction for .txt documents from raw bytes (no disk I/O).
def extract_txt_from_bytes(data: bytes) -> str:
    """Extract text from a .txt document stored as bytes (no disk I/O)."""
    try:
        content = data.decode("utf-8", errors="replace")
    except Exception as e:
        return f"Error reading the document: {e}"

    if len(content) > MAX_CHARS:
        content = (
            content[:MAX_CHARS]
            + f"\n\n[... content truncated, only the first "
              f"{MAX_CHARS} characters are shown ...]"
        )
    return content

