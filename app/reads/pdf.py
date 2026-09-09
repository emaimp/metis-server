import io

from pypdf import PdfReader

from app.core.settings import MAX_CHARS


def extract_pdf_from_bytes(data: bytes, max_pages: int = 30) -> str:
    """Extract text from a PDF stored as bytes (no disk I/O), reading the first max_pages pages."""
    try:
        reader = PdfReader(io.BytesIO(data))
        total_pages = len(reader.pages)
        pages_to_read = min(max_pages, total_pages)

        parts = []
        for page in reader.pages[:pages_to_read]:
            text = page.extract_text() or ''
            parts.append(text.strip())

        content = '\n\n'.join(p for p in parts if p)

        if total_pages > max_pages:
            content += (
                f"\n\n[... document is longer, only the first {max_pages} "
                f"pages of {total_pages} were read ...]"
            )

        if not content.strip():
            return (
                "Error: Could not extract text from the PDF. "
                "It may be a scanned document without a text layer."
            )

        return content
    except Exception as e:
        return f"Error reading the PDF: {e}"

