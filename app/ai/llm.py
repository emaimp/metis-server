import json

from app.ai.providers import llamacpp
from app.core.settings import LANGUAGE_CODE


def _language_instruction(code: str) -> str:
    """Generic instruction telling the model to respond in a given language.

    Uses the ISO 639-1 code so no localized dictionary is required.
    """
    return (
        f"Respond exclusively in the language with ISO 639-1 code '{code}'."
    )


def resolve_model(selected: str | None = None) -> str:
    """Return the model chosen by the client (UI).

    There is no default model: the user always picks one, typically populated
    from ``GET /models``. Raises ``RuntimeError`` when none is provided (the
    router maps it to HTTP 500).
    """
    model = (selected or '').strip()
    if not model:
        raise RuntimeError(
            'No model selected. Choose a model from the client (GET /models).'
        )
    return model


def generate_json_field(
    field: str, system: str, user: str, model: str,
    images: list[bytes] | None = None,
) -> dict:
    """Runs a two-attempt JSON generation in the configured language.

    The first attempt is followed by a reminder to use the configured language,
    and the second attempt is returned (unchanged behavior).
    """
    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': user},
    ]

    for attempt in range(2):
        response_text = llamacpp.chat(
            model=model,
            messages=messages,
            images=images,
            json_mode=True,
        )
        data = json.loads(response_text)
        if attempt == 0:
            messages.append({'role': 'assistant', 'content': response_text})
            messages.append({
                'role': 'user',
                'content': (
                    f'The "{field}" value must be in the configured language. '
                    f'{_language_instruction(LANGUAGE_CODE)} '
                    f'Reply again with ONLY the JSON field "{field}".'
                ),
            })

    return data


def ask_chat(
    message: str,
    model: str | None = None,
    image: bytes | None = None,
) -> str:
    """Send a message (with an optional image) to the model and return its response.

    When a document is attached, its extracted text is already embedded in the
    message by the caller.
    """
    effective = resolve_model(model)

    messages = [
        {'role': 'system', 'content': _language_instruction(LANGUAGE_CODE)},
        {'role': 'user', 'content': message},
    ]
    return llamacpp.chat(
        model=effective,
        messages=messages,
        images=[image] if image is not None else None,
    )


def ask_chat_stream(
    message: str,
    model: str | None = None,
    image: bytes | None = None,
):
    effective = resolve_model(model)

    messages = [
        {'role': 'system', 'content': _language_instruction(LANGUAGE_CODE)},
        {'role': 'user', 'content': message},
    ]
    return llamacpp.chat_stream(
        model=effective,
        messages=messages,
        images=[image] if image is not None else None,
    )


def list_models() -> list[str]:
    """Return the model names available in the active server (GET /models)."""
    return llamacpp.list_models()
