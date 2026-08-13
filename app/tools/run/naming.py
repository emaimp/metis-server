import re
from pathlib import Path

from app.ai.ollama import generate_file_name
from app.core.settings import MAX_NAME_LENGTH, TOOL_BY_EXTENSION
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
        data = generate_file_name(content, instruction)
    except Exception as e:
        return f"Error generating the file name: {str(e)}"

    name = data.get('new_name', '')
    if not isinstance(name, str) or not name.strip():
        return 'Error: Could not generate a file name.'

    return _sanitize_name(name)
