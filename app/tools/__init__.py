from app.tools.reads.docx import read_docx
from app.tools.reads.pdf import read_pdf
from app.tools.reads.txt import read_txt

available_tools = {
    'read_txt': read_txt,
    'read_pdf': read_pdf,
    'read_docx': read_docx,
}

# List of tools to register with Ollama
TOOLS = list(available_tools.values())

__all__ = ['TOOLS', 'available_tools']
