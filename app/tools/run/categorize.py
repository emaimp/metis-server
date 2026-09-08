import re

from app.ai.ollama import _generate_json_field, _language_instruction, resolve_model
from app.core.settings import (
    DEFAULT_CATEGORY,
    IMAGE_EXTENSIONS,
    LANGUAGE_CODE,
    MAX_CATEGORY_LENGTH,
    MIN_UNIQUE_WORDS,
)
from app.reads.docx import extract_docx_from_bytes
from app.reads.pdf import extract_pdf_from_bytes
from app.reads.txt import extract_txt_from_bytes

_EXTRACT_BY_EXT = {
    '.txt': extract_txt_from_bytes,
    '.pdf': extract_pdf_from_bytes,
    '.docx': extract_docx_from_bytes,
}


def _has_min_content(content: str | None) -> bool:
    """
    Returns True when the document content carries enough meaningful text to base a categorization decision on.
    """
    if not content:
        return False
    words = re.findall(r'\w+', content.lower())
    return len(set(words)) >= MIN_UNIQUE_WORDS


def _sanitize_category(name: str) -> str:
    """
    Cleans the suggested category so it is valid as a folder name.
    Keeps only the first word.
    """
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', name)
    name = re.sub(r'\s+', '_', name)
    name = name.strip(' _.')
    name = re.split(r'[\s_]+', name, maxsplit=1)[0]
    name = name[:MAX_CATEGORY_LENGTH].rstrip('_. ')
    return name.lower()


def _normalized_extension(extension: str) -> str:
    ext = extension.lower()
    return ext if ext.startswith(".") else f".{ext}"


def categorize_file(
    data: bytes,
    extension: str,
    existing_categories: list[str] | None = None,
    instruction: str = '',
    model: str | None = None,
) -> str:
    """
    Analyzes an uploaded file (as bytes) and returns its category. The frontend
    is responsible for creating the category folder and moving the file into it.

    Categorization is 100% content-based; the file name is not considered.

    Supported formats: .txt, .pdf, .docx and common image formats.

    Args:
        data: The raw bytes of the file to analyze.
        extension: The file extension (with or without leading dot).
        existing_categories: Category names the frontend already has,
            so the model can reuse them for consistency.
        instruction: Additional user requirement steering the category.
        model: Override the Ollama model for this call.

    Returns:
        The category (folder name) on success. When the document does not
        provide enough information to decide, returns the sentinel
        DEFAULT_CATEGORY ('uncategorized'). Real failures (unsupported type,
        unreadable content) return an 'Error: ...' string.
    """
    ext = _normalized_extension(extension)
    is_image = ext in IMAGE_EXTENSIONS

    if not is_image and ext not in _EXTRACT_BY_EXT:
        return f"Error: Unsupported file extension '{ext}'."

    if is_image:
        image_data = data
        content = None
    else:
        content = _EXTRACT_BY_EXT[ext](data)
        if content.startswith('Error'):
            return content
        image_data = None

    if not is_image and not _has_min_content(content):
        return DEFAULT_CATEGORY

    try:
        source = 'image' if is_image else 'document'
        system = (
            f'You are a {source} classifier. Analyze the content of the '
            f'{source} and determine its category (topic/type). '
            'Respond ONLY in JSON with the field "category": a SINGLE WORD, '
            'without extension, spaces, or underscores. Always use the same, '
            'consistent category across documents. '
            'Base the decision solely on the content. '
            f"If the content does not provide enough information to identify "
            f"a meaningful topic, do NOT guess: respond ONLY with "
            f'{{"insufficient_info": true}}. '
            f"The 'category' value MUST be a single word in the configured "
            f"language ('{LANGUAGE_CODE}'), even when the document content or "
            f'existing categories are in another language. '
            f'{_language_instruction(LANGUAGE_CODE)}'
        )
        if existing_categories:
            system += (
                ' Only reuse an existing category when it is already in the '
                f"configured language ('{LANGUAGE_CODE}') and its meaning fits; "
                'otherwise generate a new category in that language. '
                'Do not translate existing names. Existing categories: '
                f"{', '.join(existing_categories)}."
            )
        if instruction:
            system += f" Additional user requirement: {instruction}."

        user = f'Content:\n{content}\n' if content else ''

        data = _generate_json_field(
            'category', system, user, resolve_model(model),
            images=[image_data] if image_data else None,
        )
    except Exception as e:
        return f'Error generating the category: {str(e)}'

    if data.get('insufficient_info'):
        return DEFAULT_CATEGORY

    category = data.get('category', '')
    if not isinstance(category, str) or not category.strip():
        return DEFAULT_CATEGORY

    sanitized = _sanitize_category(category)
    return sanitized or DEFAULT_CATEGORY
