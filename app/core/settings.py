import os
from dotenv import load_dotenv

load_dotenv()

# LLM backend: llama.cpp OpenAI-compatible server.
LLAMA_CPP_BASE_URL = os.getenv('LLAMACPP_BASE_URL', '').strip()

# Timeout (seconds) for HTTP requests to the llama.cpp server.
LLAMA_CPP_TIMEOUT = 180

# SearXNG web search (local Docker instance).
SEARXNG_BASE_URL = os.getenv('SEARXNG_BASE_URL', 'http://localhost:8888').strip()

# Timeout (seconds) for HTTP requests to the SearXNG server.
SEARXNG_TIMEOUT = 15

# Maximum number of search results injected as context for the model.
MAX_SEARCH_RESULTS = 5

# Character cap applied to each search result snippet.
SEARCH_SNIPPET_CHARS = 300

# PIL image format names -> MIME, used to build data URIs for vision requests.
IMAGE_FORMAT_TO_MIME = {
    'PNG': 'image/png',
    'JPEG': 'image/jpeg',
    'GIF': 'image/gif',
    'WEBP': 'image/webp',
    'BMP': 'image/bmp',
}

# MIME used when the image format cannot be sniffed.
IMAGE_FORMAT_FALLBACK_MIME = 'image/png'

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
    '.wav': 'audio/wav',
    '.mp3': 'audio/mpeg',
}

# Reference-voice uploads for voice cloning (OmniVoice `ref_audio`).
# Stored and forwarded as-is: the model loads both formats natively.
VOICE_REFERENCE_EXTENSIONS = {'.wav', '.mp3'}
