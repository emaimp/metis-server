from __future__ import annotations

# Only one attribute can be selected per category (categories can be freely combined)
VOICE_CATEGORIES: dict[str, dict[str, list[str]]] = {
    "gender": {
        "label": "Gender",
        "attributes": ["male", "female"],
    },
    "age": {
        "label": "Age",
        "attributes": ["child", "teenager", "young adult", "middle-aged", "elderly"],
    },
    "pitch": {
        "label": "Pitch",
        "attributes": ["very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch"],
    },
    "style": {
        "label": "Style",
        "attributes": ["whisper"],
    },
}

# Map lowercased attribute -> category id for quick lookup and exclusivity check
_ATTRIBUTE_TO_CATEGORY: dict[str, str] = {
    attr.strip().lower(): cat_id
    for cat_id, cat in VOICE_CATEGORIES.items()
    for attr in cat["attributes"]
}

# All allowed lowercased attributes set
_ALLOWED_ATTRIBUTES: set[str] = set(_ATTRIBUTE_TO_CATEGORY.keys())


def _split_instruct(instruct: str) -> list[str]:
    # Split instruct by half-width ',' or full-width '，', strip and lower
    normalized = instruct.replace("，", ",")
    parts = [p.strip() for p in normalized.split(",")]
    # Filter empty
    return [p for p in parts if p]


def validate_instruct(instruct: str | None) -> None:
    """
    Strict validation for instruct string.

    Raises ValueError with detail if:
    - Attribute not in allowed list (any category)
    - More than one attribute from same category selected
    - Empty tokens after split (handled)

    Empty/None instruct is considered valid (no voice design).
    """
    if instruct is None or not instruct.strip():
        return
    tokens = _split_instruct(instruct)
    if not tokens:
        return

    seen_categories: dict[str, str] = {}
    invalid: list[str] = []

    for token in tokens:
        key = token.lower().strip()
        # Normalize multiple spaces
        key = " ".join(key.split())
        cat = _ATTRIBUTE_TO_CATEGORY.get(key)
        if cat is None:
            invalid.append(token)
            continue
        if cat in seen_categories:
            raise ValueError(
                f"Multiple attributes from category '{cat}' not allowed: "
                f"'{seen_categories[cat]}' and '{token}'. Only one per category."
            )
        seen_categories[cat] = token

    if invalid:
        allowed_examples = sorted(_ALLOWED_ATTRIBUTES)[:5]
        raise ValueError(
            f"Invalid instruct attribute(s): {', '.join(repr(x) for x in invalid)}. "
            f"Allowed attributes include: {', '.join(allowed_examples)}..."
        )
