import os

from app.core.settings import MAX_CHARS


# Tool definition for .txt documents
def read_txt(file_path: str) -> str:
    """
    Reads and extracts the content of a text document (.txt) for general analysis.

    Args:
        file_path: The path to the text file to read.
    """
    if not os.path.exists(file_path):
        return f"Error: The file '{file_path}' does not exist."

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return f"Error reading the document: {str(e)}"

    if len(content) > MAX_CHARS:
        content = (
            content[:MAX_CHARS]
            + f"\n\n[... content truncated, only the first "
              f"{MAX_CHARS} characters are shown ...]"
        )

    return content
