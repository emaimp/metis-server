import io
import os

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


def extract_pdf_text(file_path: str, max_pages: int) -> str:
    """
    Extracts text from a PDF (only the first max_pages pages).

    Args:
        file_path: The path to the PDF file.
        max_pages: Maximum number of pages to read.
    """
    if not os.path.exists(file_path):
        return f"Error: The file '{file_path}' does not exist."

    try:
        with open(file_path, 'rb') as f:
            content = extract_pdf_from_bytes(f.read(), max_pages)
    except Exception as e:
        return f"Error reading the PDF: {str(e)}"

    if len(content) > MAX_CHARS:
        content = (
            content[:MAX_CHARS]
            + f"\n\n[... content truncated, only the first "
              f"{MAX_CHARS} characters are shown ...]"
        )

    return content


# Tool definition for .pdf documents
def read_pdf(file_path: str) -> str:
    """
    Reads and extracts the content of a PDF document for general analysis.

    Args:
        file_path: The path to the PDF file to read.
    """
    content = extract_pdf_text(file_path, 30)  # first 30 pages
    if content.startswith('Error'):
        return content

    if len(content) > MAX_CHARS:
        content = (
            content[:MAX_CHARS]
            + f"\n\n[... content truncated, only the first "
              f"{MAX_CHARS} characters are shown ...]"
        )

    return content
