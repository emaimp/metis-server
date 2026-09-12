import re

from app.ai.llm import generate_json_field, _language_instruction, resolve_model
from app.core.settings import (
    DEFAULT_NAME,
    IMAGE_EXTENSIONS,
    LANGUAGE_CODE,
    MAX_NAME_LENGTH,
    MAX_NAME_WORDS,
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
    Returns True when the document content carries enough meaningful text to base a naming decision on.
    """
    if not content:
        return False
    words = re.findall(r'\w+', content.lower())
    return len(set(words)) >= MIN_UNIQUE_WORDS


def _sanitize_name(name: str) -> str:
    """
    Cleans the suggested name so it is valid as a file name, keeping at most MAX_NAME_WORDS and MAX_NAME_LENGTH.
    """
    # Invalid characters in file systems and control characters
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', name)
    # Collapses whitespace into underscores
    name = re.sub(r'\s+', '_', name)
    # Removes an accidental trailing extension (e.g. ".txt")
    name = re.sub(r'\.\w{1,5}$', '', name)
    # Trims extra separators and whitespace
    name = name.strip(' _.')
    # Keeps only the first MAX_NAME_WORDS words
    words = [w for w in name.split('_') if w][:MAX_NAME_WORDS]
    name = '_'.join(words)
    # Emergency cap against degenerate output or instruction abuse
    return name[:MAX_NAME_LENGTH]


def _filter_existing_by_extension(
    target_ext: str, existing_files: list[str] | None
) -> list[str] | None:
    """Return only names whose extension matches the target file exactly."""
    if not existing_files:
        return None
    filtered = [f for f in existing_files if f.lower().endswith(target_ext)]
    return filtered or None


def _normalized_extension(extension: str) -> str:
    ext = extension.lower()
    return ext if ext.startswith(".") else f".{ext}"


def rename_file(
    data: bytes,
    extension: str,
    model: str | None = None,
    existing_files: list[str] | None = None,
    instruction: str = '',
) -> str:
    """
    Analyzes the content of an uploaded file (as bytes) and returns the new
    file name that the client will apply.

    When the document does not provide enough information to decide,
    returns the sentinel DEFAULT_NAME ('unassigned') instead of a guess.

    Supported formats: .txt, .pdf, .docx and common image formats.

    Args:
        data: The raw bytes of the file to analyze.
        extension: The file extension (with or without leading dot).
        model: Model chosen by the client for this call (via resolve_model).
        existing_files: Names already present in the target folder; only
            those with the same extension as the analyzed file are
            considered to avoid duplicates.
        instruction: Additional user requirement for the new name.
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
        return DEFAULT_NAME

    try:
        source = 'image' if is_image else 'document'
        system = (
            f'Analyze the {source} and generate a descriptive, '
            'short file name to rename it. Respond ONLY in JSON with the field '
            "'new_name', without extension, using at most "
            f'{MAX_NAME_WORDS} words separated by underscores. '
            f"The 'new_name' value MUST be in the configured language "
            f"('{LANGUAGE_CODE}'). {_language_instruction(LANGUAGE_CODE)} "
            'If the content does not provide enough information to infer a '
            'meaningful name (empty text, illegible image, no discernible '
            'subject), do NOT invent one: respond ONLY with '
            '{"insufficient_info": true}.'
        )
        if instruction:
            system += f" Additional user requirement: {instruction}."
        filtered = _filter_existing_by_extension(ext, existing_files)
        if filtered:
            system += (
                ' The following file names already exist for this file type; '
                'generate a DIFFERENT name that does not duplicate any of them: '
                f"{', '.join(filtered)}."
            )

        user = content if content else ''
        data = generate_json_field(
            'new_name', system, user, resolve_model(model),
            images=[image_data] if image_data else None,
        )
    except Exception as e:
        return f"Error generating the file name: {str(e)}"

    if data.get('insufficient_info'):
        return DEFAULT_NAME

    name = data.get('new_name', '')
    if not isinstance(name, str) or not name.strip():
        return DEFAULT_NAME

    sanitized = _sanitize_name(name)
    return sanitized or DEFAULT_NAME
