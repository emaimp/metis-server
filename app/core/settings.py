import os
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv('OLLAMA_MODEL', '')

MAX_CHARS = 30000

MAX_NAME_LENGTH = 30

MAX_CATEGORY_LENGTH = 30

ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.docx'}

TOOL_BY_EXTENSION = {
    '.txt': 'read_txt',
    '.pdf': 'read_pdf',
    '.docx': 'read_docx',
}
