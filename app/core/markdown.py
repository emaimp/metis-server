from __future__ import annotations

import re


def strip_markdown(md: str) -> str:
    """
    Convert markdown to plain spoken text.

    Rules (per spec):
    - Code blocks ```...``` / ~~~...~~~ : omitted completely
    - Tables (|): omitted completely
    - Links [text](url): keep only text
    - Images ![alt](url): keep only alt
    - Inline code `code`: keep inner text without backticks
    - Headings, bold, italic, strike, blockquotes, lists: keep inner text
    """
    if not md or not md.strip():
        return ""

    text = md

    # 1. Code blocks (```...``` and ~~~...~~~) - omitted
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"~~~.*?~~~", "", text, flags=re.DOTALL)

    # 2. Inline code `...` -> keep inner text
    text = re.sub(r"`([^`]+?)`", r"\1", text)

    # 3. Images ![alt](url) -> alt
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)

    # 4. Links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # 5. Autolinks <https://...>
    text = re.sub(r"<https?://[^>]+>", "", text)

    # 6. Bare URLs https://... -> removed (avoid speaking URLs)
    text = re.sub(r"https?://\S+", "", text)

    # 7. HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # 8. Headings: ## Title -> Title
    text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.MULTILINE)

    # 9. Blockquotes: > text -> text
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)

    # 10. Horizontal rules: ---, ***, ___
    text = re.sub(r"^\s*([-*_])\s*(\1\s*){2,}\s*$", "", text, flags=re.MULTILINE)

    # 11. Tables: omit any line containing '|'
    lines = []
    for line in text.splitlines():
        if "|" in line:
            continue
        lines.append(line)
    text = "\n".join(lines)

    # 12. Bold / italic / strike - keep inner text
    # Order matters: longer delimiters first
    text = re.sub(r"\*\*\*([^*]+?)\*\*\*", r"\1", text)
    text = re.sub(r"\*\*([^*]+?)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+?)__", r"\1", text)
    text = re.sub(r"~~([^~]+?)~~", r"\1", text)
    text = re.sub(r"\*([^*]+?)\*", r"\1", text)
    text = re.sub(r"_([^_]+?)_", r"\1", text)

    # 13. Unordered list markers: - , * , +
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    # Task list: - [x] / - [ ]
    text = re.sub(r"^\s*[-*]\s*\[[ xX]\]\s*", "", text, flags=re.MULTILINE)
    # Ordered list: 1. 2. ...
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # 14. Footnote refs [^1]
    text = re.sub(r"\[\^[^\]]+\]", "", text)

    # 15. Reference-style link definitions [1]: url - remove
    text = re.sub(r"^\s*\[[^\]]+\]:\s*\S+.*$", "", text, flags=re.MULTILINE)

    # 16. Collapse whitespace and newlines for prosody
    # Normalize multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Replace single newlines (not double) with space
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Collapse double newlines to space as well (paragraphs -> sentence gap)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


def prepare_tts_text(md: str) -> str:
    # Alias with more explicit TTS intent
    return strip_markdown(md)
