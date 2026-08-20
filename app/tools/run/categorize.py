import re
from pathlib import Path

from app.ai.ollama import _generate_json_field, _language_instruction, resolve_model
from app.core.settings import IMAGE_EXTENSIONS, LANGUAGE_CODE, MAX_CATEGORY_LENGTH, TOOL_BY_EXTENSION
from app.tools import available_tools
from app.tools.reads.image import read_image_bytes


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
    return name.lower() or 'uncategorized'


def categorize_file(
    file_path: str,
    existing_categories: list[str] | None = None,
    model: str | None = None,
) -> str:
    """
    Analyzes a file and returns its category. The frontend is responsible
    for creating the category folder and moving the file into it.

    The file content is always analyzed; the file name is only used as
    an additional hint for the model.

    Supported formats: .txt, .pdf, .docx and common image formats.

    Args:
        file_path: The path to the document to categorize.
        existing_categories: Category names the frontend already has,
            so the model can reuse them for consistency.
        model: Override the Ollama model for this call.

    Returns:
        The category (folder name) on success, or an 'Error: ...' string.
    """
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        return f"Error: The file '{file_path}' does not exist."

    ext = path.suffix.lower()
    is_image = ext in IMAGE_EXTENSIONS

    if not is_image and ext not in TOOL_BY_EXTENSION:
        return f"Error: Unsupported file extension '{ext}' for categorization."

    stem = path.stem
    try:
        if is_image:
            image_data = read_image_bytes(str(path))
            content = None
        else:
            reader = available_tools.get(TOOL_BY_EXTENSION[ext])
            content = reader(str(path))
            if content.startswith('Error'):
                return content
            image_data = None
    except Exception as e:
        return f"Error reading the file: {e}"

    try:
        source = 'image' if is_image else 'document'
        system = (
            f'You are a {source} classifier. Analyze the content of the '
            f'{source} and determine its category (topic/type). '
            'Respond ONLY in JSON with the field "category": a SINGLE WORD, '
            'without extension, spaces, or underscores. Always use the same, '
            'consistent category across documents. '
            "Never respond 'unknown' or 'uncategorized': always pick the closest "
            f'meaningful topic based on the content. The file name is only a hint. '
            f"The 'category' value MUST be a single word in the configured "
            f"language ('{LANGUAGE_CODE}'), even when the document content or "
            f'existing categories are in another language. '
            f'{_language_instruction(LANGUAGE_CODE)}'
        )
        if existing_categories:
            system += (
                ' Reuse the concept of an existing category only when it fits, '
                'but ALWAYS output the "category" value in the configured '
                'language, translating the existing name if needed. Existing '
                'categories (possibly in another language): '
                f"{', '.join(existing_categories)}."
            )

        user = f'File name: {stem}\n'
        if content:
            user += f'Content:\n{content}\n'
        elif not is_image:
            user += 'No content provided; classify based on the file name only.'

        data = _generate_json_field(
            'category', system, user, resolve_model(model),
            images=[image_data] if image_data else None,
        )
    except Exception as e:
        return f'Error generating the category: {str(e)}'

    category = data.get('category', '')
    if not isinstance(category, str) or not category.strip():
        return 'Error: Could not generate a category.'

    return _sanitize_category(category)
