from app.reads.docx import read_docx
from app.reads.pdf import read_pdf
from app.reads.txt import read_txt

available_tools = {
    'read_txt': read_txt,
    'read_pdf': read_pdf,
    'read_docx': read_docx,
}

# List of tools to register with Ollama
TOOLS = list(available_tools.values())

__all__ = ['TOOLS', 'available_tools']
