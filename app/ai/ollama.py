import json

from ollama import chat, list as ollama_list

from app.core.settings import LANGUAGE_CODE, MODEL


def _language_instruction(code: str) -> str:
    """
    Generic instruction telling the model to respond in a given language.
    Uses the ISO 639-1 code so no localized dictionary is required.
    """
    return (
        f"Respond exclusively in the language with ISO 639-1 code '{code}'."
    )


def resolve_model(selected: str | None = None) -> str:
    """
    Resolve the effective Ollama model name.

    Precedence:
      1. Explicitly selected model (from the client/UI).
      2. OLLAMA_MODEL env var.
      3. Clear error if none of the above.

    Raises:
        RuntimeError: if no model can be resolved.
    """
    model = selected or MODEL
    if not model:
        raise RuntimeError(
            'No model configured. Set OLLAMA_MODEL in the .env file '
            'or select a model from the client.'
        )
    return model


def _generate_json_field(
    field: str, system: str, user: str, model: str,
    images: list[bytes] | None = None,
) -> dict:
    """
    Runs a two-attempt JSON generation. The first attempt is followed by a
    reminder to use the configured language, and the second attempt is returned.
    """
    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': user},
    ]
    if images:
        messages[1]['images'] = images

    for attempt in range(2):
        response = chat(
            model=model,
            messages=messages,
            format='json',
            think=False,
            stream=False,
        )
        data = json.loads(response.message.content)
        if attempt == 0:
            messages.append({'role': 'assistant', 'content': response.message.content})
            messages.append({
                'role': 'user',
                'content': (
                    f'The "{field}" value must be in the configured language. '
                    f'{_language_instruction(LANGUAGE_CODE)} '
                    f'Reply again with ONLY the JSON field "{field}".'
                ),
            })

    return data


def list_models() -> list[str]:
    """Return the list of installed Ollama model names."""
    try:
        response = ollama_list()
    except Exception as e:
        raise ConnectionError(f'Cannot connect to Ollama: {e}') from e
    return [m.model for m in response.models]


def ask_chat(
    message: str,
    image: bytes | None = None,
    model: str | None = None,
) -> str:
    """
    Sends a message (with an optional image) to the model and returns its response.

    Args:
        message: The user's message. When a document is attached, its extracted
            text is already embedded in the message by the caller.
        image: The bytes of an optional image to analyze.
        model: The Ollama model to use (resolved by caller or auto-resolved).
    """
    effective = resolve_model(model)

    messages = [
        {'role': 'system', 'content': _language_instruction(LANGUAGE_CODE)}
    ]

    user_message = {
        'role': 'user',
        'content': message,
    }
    if image is not None:
        user_message['images'] = [image]
    messages.append(user_message)

    response = chat(
        model=effective,
        messages=messages,
        think=False,
        stream=False,
    )

    return response.message.content
