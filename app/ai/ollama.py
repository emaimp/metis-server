import json
from collections.abc import Callable
from contextvars import ContextVar

from ollama import chat, list as ollama_list

from app.core.settings import LANGUAGE_CODE, MODEL

_current_model: ContextVar[str] = ContextVar('_current_model', default='')


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
      2. Active model in the current context (set by ask_chat during tool calls).
      3. OLLAMA_MODEL env var.
      4. Clear error if none of the above.

    Raises:
        RuntimeError: if no model can be resolved.
    """
    model = selected or _current_model.get() or MODEL
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
    available_tools: dict[str, Callable] | None = None,
    tool_hint: str | None = None,
    model: str | None = None,
    return_tool_results: bool = False,
) -> str | tuple[str, dict[str, str]]:
    """
    Sends a message (with optional image and/or tools) to the model
    and returns its response.

    Args:
        message: The user's message.
        image: The bytes of an optional image to analyze.
        available_tools: Mapping of name -> function for the available tools.
        tool_hint: Additional hint (e.g. the path of an attached document)
            so the model knows it must use a tool.
        model: The Ollama model to use (resolved by caller or auto-resolved).
        return_tool_results: If True, also returns the last result of each
            tool call so far in a dict {tool_name: result}.
    """
    effective = resolve_model(model)
    token = _current_model.set(effective)

    messages = [
        {'role': 'system', 'content': _language_instruction(LANGUAGE_CODE)}
    ]
    if tool_hint:
        messages.append({'role': 'system', 'content': tool_hint})

    user_message = {
        'role': 'user',
        'content': message,
    }
    if image is not None:
        user_message['images'] = [image]
    messages.append(user_message)

    tools = list(available_tools.values()) if available_tools else None

    tool_results: dict[str, str] = {}

    response = chat(
        model=effective,
        messages=messages,
        tools=tools,
        think=False,
        stream=False,
    )

    try:
        for _ in range(5):  # max tool-call iterations
            if not response.message.tool_calls:
                break

            messages.append(response.message)
            for tool_call in response.message.tool_calls:
                function_name = tool_call.function.name
                function_args = tool_call.function.arguments
                if available_tools and function_name in available_tools:
                    result = available_tools[function_name](**function_args)
                    tool_results[function_name] = result
                    messages.append({
                        'role': 'tool',
                        'content': result,
                        'name': function_name,
                    })

            response = chat(
                model=effective,
                messages=messages,
                tools=tools,
                think=False,
                stream=False,
            )
    finally:
        _current_model.reset(token)

    if return_tool_results:
        return response.message.content, tool_results

    return response.message.content
