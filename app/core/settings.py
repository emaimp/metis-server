import os
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv('OLLAMA_MODEL', '')

LANGUAGE = os.getenv('APP_LANGUAGE', '').strip().lower()

SUPPORTED_LANGUAGES = ('en', 'es', 'zh', 'ja')

LANGUAGE_CODE = LANGUAGE if LANGUAGE in SUPPORTED_LANGUAGES else 'en'

MAX_CHARS = 30000

# Minimum distinct words a document must contain to be considered.
MIN_UNIQUE_WORDS = 20

# Fixed sentinel values returned when a file does not carry enough information to decide.
DEFAULT_NAME = 'unassigned'

DEFAULT_CATEGORY = 'uncategorized'

# Maximum words a generated file name may have (underscore-separated).
MAX_NAME_WORDS = 3

# Character cap applied after the word limit; legitimate names never reach it.
MAX_NAME_LENGTH = 60

MAX_CATEGORY_LENGTH = 30

DOCUMENT_EXTENSIONS = {'.txt', '.pdf', '.docx'}

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}

# Maximum size allowed for uploaded documents/images (in bytes). 10 MB.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

MIME_BY_EXTENSION = {
    '.txt': 'text/plain',
    '.pdf': 'application/pdf',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.webp': 'image/webp',
    '.bmp': 'image/bmp',
}
