import base64
import io

import requests

from app.core import settings


def _image_data_uri(data: bytes) -> str:
    """Encode raw image bytes as a base64 data URI with a sniffed MIME type."""
    mime = settings.IMAGE_FORMAT_FALLBACK_MIME
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            mime = settings.IMAGE_FORMAT_TO_MIME.get(
                img.format, settings.IMAGE_FORMAT_FALLBACK_MIME,
            )
    except Exception:
        pass  # fall back to the default format when sniffing fails
    return f'data:{mime};base64,{base64.b64encode(data).decode()}'


def _build_messages(messages: list[dict], images: list[bytes] | None) -> list[dict]:
    """Copy OpenAI-style messages; attach images to the last user message."""
    msgs = [dict(m) for m in messages]
    if images:
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i]['role'] == 'user':
                parts = [{'type': 'text', 'text': msgs[i].get('content', '')}]
                parts += [
                    {'type': 'image_url', 'image_url': {'url': _image_data_uri(b)}}
                    for b in images
                ]
                msgs[i] = {**msgs[i], 'content': parts}
                break
    return msgs


def chat(
    model: str,
    messages: list[dict],
    images: list[bytes] | None = None,
    json_mode: bool = False,
) -> str:
    """Send OpenAI-style messages to a llama.cpp server and return the text."""
    base_url = settings.LLAMA_CPP_BASE_URL
    if not base_url:
        raise ConnectionError('LLAMACPP_BASE_URL is not configured')

    payload = {
        'model': model,
        'messages': _build_messages(messages, images),
        'stream': False,
    }
    if json_mode:
        payload['response_format'] = {'type': 'json_object'}

    try:
        resp = requests.post(
            f'{base_url}/chat/completions',
            json=payload,
            timeout=settings.LLAMA_CPP_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content']
    except requests.RequestException as e:
        raise ConnectionError(f'Cannot connect to llama.cpp: {e}') from e
    except (ValueError, KeyError, TypeError) as e:
        raise RuntimeError(f'Unexpected response from llama.cpp: {e}') from e


def list_models() -> list[str]:
    """Return the model names served by the llama.cpp server (single or router mode)."""
    base_url = settings.LLAMA_CPP_BASE_URL
    if not base_url:
        raise ConnectionError('LLAMACPP_BASE_URL is not configured')
    try:
        resp = requests.get(f'{base_url}/models', timeout=settings.LLAMA_CPP_TIMEOUT)
        resp.raise_for_status()
        return [m['id'] for m in resp.json()['data']]
    except requests.RequestException as e:
        raise ConnectionError(f'Cannot connect to llama.cpp: {e}') from e
    except (ValueError, KeyError, TypeError) as e:
        raise RuntimeError(f'Unexpected response from llama.cpp: {e}') from e
