import re
from pathlib import Path

from app.ai.ollama import _generate_json_field, _language_instruction, resolve_model
from app.core.settings import LANGUAGE_CODE, MAX_NAME_LENGTH, TOOL_BY_EXTENSION
from app.tools import available_tools


def _sanitize_name(name: str) -> str:
    """
    Cleans the suggested name so it is valid as a file name.
    """
    # Invalid characters in file systems and control characters
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', name)
    # Collapses whitespace into underscores
    name = re.sub(r'\s+', '_', name)
    # Removes an accidental trailing extension (e.g. ".txt")
    name = re.sub(r'\.\w{1,5}$', '', name)
    # Trims extra separators and whitespace
    name = name.strip(' _.')
    # Limits the length
    name = name[:MAX_NAME_LENGTH].rstrip('_. ')
    return name or 'document'


def rename_tool_hint(file_path: str) -> str:
    """Hint for the chat model to invoke rename_file correctly."""
    return (
        f"The user attached a document at '{file_path}' and wants "
        "to rename it. You MUST call the tool 'rename_file' with the "
        "'file_path' argument set to that exact path. Do not reply "
        "without calling the tool. The client applies the rename with "
        "the name returned by the tool, so after calling it confirm to "
        "the user that the file has been renamed, e.g. 'The file was "
        "renamed to <name>'."
    )


def rename_file(file_path: str, instruction: str = '') -> str:
    """
    Analyzes the content of a document and returns the new file name
    that the client will apply.

    Supported formats: .txt, .pdf, .docx (via TOOL_BY_EXTENSION).

    Args:
        file_path: The path to the document to rename.
        instruction: Additional user requirement for the new name.
    """
    ext = Path(file_path).suffix.lower()
    tool_name = TOOL_BY_EXTENSION.get(ext)
    reader = available_tools.get(tool_name) if tool_name else None
    if reader is None:
        return f"Error: Unsupported file extension '{ext}' for renaming."

    content = reader(file_path)
    if content.startswith('Error'):
        return content

    try:
        system = (
            'Analyze the content of the document and generate a descriptive, '
            'short file name to rename it. Respond ONLY in JSON with the field '
            "'new_name', without extension, using underscores instead of spaces. "
            f"The 'new_name' value MUST be in the configured language "
            f"('{LANGUAGE_CODE}'). {_language_instruction(LANGUAGE_CODE)}"
        )
        if instruction:
            system += f" Additional user requirement: {instruction}."

        data = _generate_json_field('new_name', system, content, resolve_model())
    except Exception as e:
        return f"Error generating the file name: {str(e)}"

    name = data.get('new_name', '')
    if not isinstance(name, str) or not name.strip():
        return 'Error: Could not generate a file name.'

    return _sanitize_name(name)
