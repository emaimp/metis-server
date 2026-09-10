from app.reads.docx import extract_docx_from_bytes
from app.reads.pdf import extract_pdf_from_bytes
from app.reads.txt import extract_txt_from_bytes

_EXTRACT_BY_EXT = {
    '.txt': extract_txt_from_bytes,
    '.pdf': extract_pdf_from_bytes,
    '.docx': extract_docx_from_bytes,
}


def extract_document_text(data: bytes, extension: str) -> str:
    """Extract text from a document (as bytes) by extension; .txt, .pdf, .docx.

    Returns an 'Error: ...' string on unsupported extension, like the action tools.
    """
    ext = extension.lower()
    ext = ext if ext.startswith(".") else f".{ext}"
    extractor = _EXTRACT_BY_EXT.get(ext)
    if extractor is None:
        return f"Error: Unsupported file extension '{ext}'."
    return extractor(data)
