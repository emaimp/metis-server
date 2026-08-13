import json
from collections.abc import Callable
from contextvars import ContextVar

from ollama import chat, list as ollama_list

from app.core.settings import MODEL

_current_model: ContextVar[str] = ContextVar('_current_model', default='')


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

    messages = []
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

    _current_model.reset(token)

    if return_tool_results:
        return response.message.content, tool_results

    return response.message.content


def generate_file_name(content: str, instruction: str = '') -> dict:
    """
    Asks the model for a suggested file name for a given content.

    Args:
        Content: The content of the document to analyze.
        Instruction: Additional user requirement for the suggestion.
    """
    effective = _current_model.get() or resolve_model(None)

    system = (
        'Analyze the content of the document and generate a descriptive, '
        'short file name to rename it. Respond ONLY in JSON with the field '
        "'new_name', without extension, using underscores instead of spaces."
    )
    if instruction:
        system += f" Additional user requirement: {instruction}."

    response = chat(
        model=effective,
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': content},
        ],
        format='json',
        think=False,
        stream=False,
    )

    return json.loads(response.message.content)
