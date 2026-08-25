import os
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv('OLLAMA_MODEL', '')

LANGUAGE = os.getenv('APP_LANGUAGE', '').strip().lower()

SUPPORTED_LANGUAGES = ('en', 'es', 'zh', 'ja')

LANGUAGE_CODE = LANGUAGE if LANGUAGE in SUPPORTED_LANGUAGES else 'en'

MAX_CHARS = 30000

# Minimum alphanumeric characters a document must have to be considered.
MIN_CONTENT_CHARS = 50

# Fixed sentinel values returned when a file does not carry enough information to decide.
DEFAULT_NAME = 'unassigned'

DEFAULT_CATEGORY = 'uncategorized'

MAX_NAME_LENGTH = 30

MAX_CATEGORY_LENGTH = 30

DOCUMENT_EXTENSIONS = {'.txt', '.pdf', '.docx'}

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}

TOOL_BY_EXTENSION = {
    '.txt': 'read_txt',
    '.pdf': 'read_pdf',
    '.docx': 'read_docx',
    '.png': 'read_image',
    '.jpg': 'read_image',
    '.jpeg': 'read_image',
    '.gif': 'read_image',
    '.webp': 'read_image',
    '.bmp': 'read_image',
}
