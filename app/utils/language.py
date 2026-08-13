import re

# ---------------------------------------------------------------------------
# Language registry: ISO 639-1 code → configuration
#   name        – human-readable name
#   words       – frozenset of stopwords / common function words
#   instruction – localized prompt so the model responds in this language
# ---------------------------------------------------------------------------
LANGUAGES: dict[str, dict] = {
    'en': {
        'name': 'English',
        'words': frozenset({
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'have', 'has',
            'do', 'does', 'can', 'could', 'would', 'should', 'will',
            'this', 'that', 'these', 'those', 'it', 'its', 'my', 'your',
            'file', 'document', 'rename', 'name', 'read', 'and', 'or',
            'but', 'not', 'with', 'for', 'in', 'on', 'at', 'to', 'of',
            'be', 'if', 'so', 'as', 'no', 'yes', 'i', 'you', 'he', 'she',
            'we', 'they', 'me', 'him', 'us', 'them', 'what', 'how',
            'when', 'where', 'why', 'which', 'who', 'please', 'thanks',
        }),
        'instruction': 'The user wrote in English. You MUST respond in English.',
    },
    'es': {
        'name': 'Spanish',
        'words': frozenset({
            'el', 'la', 'los', 'las', 'un', 'una', 'de', 'del', 'en',
            'es', 'está', 'que', 'por', 'con', 'para', 'no', 'sí',
            'este', 'esta', 'como', 'pero', 'más', 'muy', 'hay', 'tiene',
            'puedo', 'puedes', 'hola', 'gracias', 'archivo', 'documento',
            'renombrar', 'cambiar', 'nombre', 'su', 'sus', 'me', 'te',
            'lo', 'al', 'se', 'y', 'o', 'e', 'a', 'tu', 'mi', 'nos',
            'les', 'ellos', 'eso', 'ese', 'aquí', 'ahora', 'también',
            'cuando', 'donde', 'porque', 'cómo', 'qué', 'quién',
        }),
        'instruction': 'El usuario escribió en español. DEBES responder en español.',
    },
    'zh': {
        'name': 'Chinese',
        'words': frozenset({
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
            '都', '一', '上', '也', '很', '到', '说', '要', '去', '你',
            '会', '着', '没有', '看', '好', '自己', '这', '他', '她',
            '们', '您', '请', '谢谢', '文件', '文档', '重命名', '名称',
            '读取', '分析', '可以', '不能', '执行', '操作', '修改',
        }),
        'instruction': '用户用中文提问。请务必用中文回答。',
    },
    'ja': {
        'name': 'Japanese',
        'words': frozenset(),
        'instruction': 'ユーザーは日本語で質問しました。必ず日本語で回答してください。',
    },
}

# Diacritics used as tie-breakers per language
_DIACRITICS: dict[str, re.Pattern] = {
    'es': re.compile(r'[ñ¿¡áéíóú]'),
}

# Script ranges for non-Latin detection (priority order matters)
_SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    'ja': [(0x3040, 0x309F), (0x30A0, 0x30FF)],  # Hiragana, Katakana
    'zh': [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],  # CJK Unified Ideographs
}

_PUNCTUATION = re.compile(r'[^\w\s]', re.UNICODE)


def _normalize(text: str) -> set[str]:
    """Lowercase text and return a set of alphanumeric tokens."""
    text = text.lower()
    text = _PUNCTUATION.sub(' ', text)
    return set(text.split())


def _score_by_script(text: str) -> str | None:
    """
    Return the language code if the text is dominated by a non-Latin script.
    Priority: Japanese (Hiragana/Katakana) > Chinese (CJK only).
    """
    # Japanese first: check unique scripts (Hiragana/Katakana)
    if _SCRIPT_RANGES.get('ja'):
        ja_count = sum(
            1 for ch in text
            if any(lo <= ord(ch) <= hi for lo, hi in _SCRIPT_RANGES['ja'])
        )
        if ja_count >= 1:
            return 'ja'

    # Chinese: CJK only (no Japanese scripts present)
    if _SCRIPT_RANGES.get('zh'):
        zh_count = sum(
            1 for ch in text
            if any(lo <= ord(ch) <= hi for lo, hi in _SCRIPT_RANGES['zh'])
        )
        if zh_count >= 2:
            return 'zh'

    return None


def _score_by_words(tokens: set[str]) -> str:
    """Score Latin-script text by stopword overlap and diacritics."""
    scores: dict[str, int] = {}
    for lang, cfg in LANGUAGES.items():
        if not cfg['words']:
            continue
        scores[lang] = len(tokens & cfg['words'])

    # Diacritic tie-breaker
    raw = ' '.join(tokens)
    for lang, pat in _DIACRITICS.items():
        if pat.search(raw):
            scores[lang] = scores.get(lang, 0) + 1

    if not scores:
        return 'en'

    best_lang = max(scores, key=scores.get)
    if scores[best_lang] == 0:
        return 'en'
    return best_lang


def detect_language(text: str) -> str:
    """
    Detect the language of a short text.

    Returns an ISO 639-1 code (e.g. 'es', 'en', 'zh', 'ja').
    Falls back to 'en' for empty or unrecognised text.
    """
    if not text or not text.strip():
        return 'en'

    # Non-Latin script check first
    script_lang = _score_by_script(text)
    if script_lang:
        return script_lang

    # Latin-script: tokenise and score
    tokens = _normalize(text)
    return _score_by_words(tokens)


def language_instruction(code: str) -> str:
    """
    Return the localized instruction sentence telling the LLM to
    respond in the given language. Falls back to English if unknown.
    """
    cfg = LANGUAGES.get(code)
    if cfg:
        return cfg['instruction']
    return LANGUAGES['en']['instruction']


def supported_languages() -> list[str]:
    """Return the list of supported ISO 639-1 codes."""
    return sorted(LANGUAGES.keys())
