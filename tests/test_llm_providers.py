import base64
import pathlib
import sys

# Ensure the project root (containing the 'app' package) is importable
# when pytest is run from anywhere.
ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
import requests

from app.ai import llm
from app.core import settings

# Minimal real 1x1 transparent PNG so the provider can sniff a format.
_TINY_PNG = base64.b64decode(
    b'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk'
    b'YAAAAAYAAjCB0C8AAAAASUVORK5CYII='
)


def _fake_resp(payload):
    """Mimics a requests.Response (raise_for_status + json)."""

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    return _Resp()


# ---------- Settings ----------

def test_settings_expose_llm_constants():
    assert settings.LLAMA_CPP_TIMEOUT == 180
    assert settings.IMAGE_FORMAT_TO_MIME['PNG'] == 'image/png'
    assert settings.IMAGE_FORMAT_TO_MIME['JPEG'] == 'image/jpeg'
    assert settings.IMAGE_FORMAT_TO_MIME['WEBP'] == 'image/webp'
    assert settings.IMAGE_FORMAT_FALLBACK_MIME == 'image/png'


# ---------- llama.cpp provider (transport) ----------

def test_llamacpp_provider_chat_payload(monkeypatch):
    from app.ai.providers import llamacpp as provider

    monkeypatch.setattr('app.core.settings.LLAMA_CPP_BASE_URL', 'http://srv:8080/v1')
    captured = {}

    def _fake_post(url, **kwargs):
        captured['url'] = url
        captured.update(kwargs)
        return _fake_resp({
            'choices': [{'message': {'content': 'resp', 'reasoning_content': 'NO USAR'}}],
        })

    monkeypatch.setattr(requests, 'post', _fake_post)

    out = provider.chat(
        model='m',
        messages=[{'role': 'user', 'content': 'q'}],
        images=[_TINY_PNG],
        json_mode=True,
    )

    assert out == 'resp'
    assert captured['url'] == 'http://srv:8080/v1/chat/completions'
    body = captured['json']
    assert body['model'] == 'm'
    assert body['stream'] is False
    assert body['response_format'] == {'type': 'json_object'}
    user_part = body['messages'][-1]['content']
    assert user_part[0] == {'type': 'text', 'text': 'q'}
    assert user_part[1]['type'] == 'image_url'
    assert user_part[1]['image_url']['url'].startswith('data:image/png;base64,')
    assert captured['timeout'] == settings.LLAMA_CPP_TIMEOUT


def test_llamacpp_provider_chat_minimal(monkeypatch):
    from app.ai.providers import llamacpp as provider

    monkeypatch.setattr('app.core.settings.LLAMA_CPP_BASE_URL', 'http://srv:8080/v1')
    captured = {}

    def _fake_post(url, **kwargs):
        captured.update(kwargs)
        return _fake_resp({'choices': [{'message': {'content': 'ok'}}]})

    monkeypatch.setattr(requests, 'post', _fake_post)
    provider.chat(model='m', messages=[{'role': 'user', 'content': 'q'}])

    body = captured['json']
    assert body['messages'][-1] == {'role': 'user', 'content': 'q'}
    assert 'response_format' not in body


class _FakeSSEResponse:
    """Mimics requests' streaming iter_lines encoding behavior exactly.

    With ``decode_unicode=True`` it decodes the lines with ISO-8859-1 (the
    fallback requests applies to any text/* without charset — the mojibake
    bug); with ``decode_unicode=False`` it yields the raw UTF-8 bytes.
    """

    def __init__(self, lines):
        self._lines = lines
        self.closed = False

    def raise_for_status(self):
        pass

    def iter_lines(self, **kwargs):
        decode = kwargs.get('decode_unicode')
        for line in self._lines:
            yield line.decode('iso-8859-1') if decode else line

    def close(self):
        self.closed = True


def test_llamacpp_chat_stream_decodes_utf8(monkeypatch):
    from app.ai.providers import llamacpp as provider

    monkeypatch.setattr('app.core.settings.LLAMA_CPP_BASE_URL', 'http://srv:8080/v1')
    fake = _FakeSSEResponse([
        'data: {"choices":[{"delta":{"content":"t"}}]}\n'.encode('utf-8'),
        'data: {"choices":[{"delta":{"content":"ú final"}}]}\n'.encode('utf-8'),
        'data: {"choices":[{"delta":{"reasoning_content":"pensando"}}]}\n'.encode('utf-8'),
        b'data: [DONE]',
    ])
    monkeypatch.setattr(requests, 'post', lambda *a, **k: fake)

    out = list(provider.chat_stream(model='m', messages=[{'role': 'user', 'content': 'q'}]))

    # Accents must arrive intact (would be "Ãº final" with the ISO-8859-1 bug)
    assert out == ['t', 'ú final']
    assert fake.closed


def test_llamacpp_provider_connection_error(monkeypatch):
    from app.ai.providers import llamacpp as provider

    monkeypatch.setattr('app.core.settings.LLAMA_CPP_BASE_URL', 'http://srv:8080/v1')

    def _boom(*a, **k):
        raise requests.ConnectionError('down')

    monkeypatch.setattr(requests, 'post', _boom)
    with pytest.raises(ConnectionError):
        provider.chat(model='m', messages=[{'role': 'user', 'content': 'q'}])


def test_llamacpp_provider_list_models(monkeypatch):
    from app.ai.providers import llamacpp as provider

    monkeypatch.setattr('app.core.settings.LLAMA_CPP_BASE_URL', 'http://srv:8080/v1')
    captured = {}

    def _fake_get(url, **kwargs):
        captured['url'] = url
        return _fake_resp({'data': [{'id': 'qwen'}, {'id': 'llama3'}]})

    monkeypatch.setattr(requests, 'get', _fake_get)
    assert provider.list_models() == ['qwen', 'llama3']
    assert captured['url'] == 'http://srv:8080/v1/models'


def test_llamacpp_provider_list_models_connection_error(monkeypatch):
    from app.ai.providers import llamacpp as provider

    monkeypatch.setattr('app.core.settings.LLAMA_CPP_BASE_URL', 'http://srv:8080/v1')

    def _boom(*a, **k):
        raise requests.ConnectionError('down')

    monkeypatch.setattr(requests, 'get', _boom)
    with pytest.raises(ConnectionError):
        provider.list_models()


def test_llamacpp_provider_without_base_url(monkeypatch):
    from app.ai.providers import llamacpp as provider

    monkeypatch.setattr('app.core.settings.LLAMA_CPP_BASE_URL', '')
    with pytest.raises(ConnectionError):
        provider.chat(model='m', messages=[{'role': 'user', 'content': 'q'}])
    with pytest.raises(ConnectionError):
        provider.list_models()


# ---------- Facade (app/ai/llm.py) ----------

def test_resolve_model_requires_client_model():
    assert llm.resolve_model('modelo-x') == 'modelo-x'
    assert llm.resolve_model('  modelo-y  ') == 'modelo-y'
    with pytest.raises(RuntimeError):
        llm.resolve_model(None)
    with pytest.raises(RuntimeError):
        llm.resolve_model('   ')


def test_ask_chat_builds_messages(monkeypatch):
    captured = {}

    def _fake_chat(**kwargs):
        captured.update(kwargs)
        return 'respuesta'

    monkeypatch.setattr('app.ai.providers.llamacpp.chat', _fake_chat)

    out = llm.ask_chat('hola', model='m', image=_TINY_PNG)
    assert out == 'respuesta'
    assert captured['model'] == 'm'
    msgs = captured['messages']
    assert msgs[0]['role'] == 'system'
    assert 'ISO 639-1 code' in msgs[0]['content']
    assert msgs[1]['content'] == 'hola'
    assert captured['images'] == [_TINY_PNG]


def test_ask_chat_requires_model():
    with pytest.raises(RuntimeError):
        llm.ask_chat('hola')


def test_generate_json_field_two_attempts(monkeypatch):
    calls = []

    def _fake_chat(**kwargs):
        calls.append(kwargs)
        return '{"category": "contenido"}'

    monkeypatch.setattr('app.ai.providers.llamacpp.chat', _fake_chat)

    out = llm.generate_json_field('category', 'sys', 'user', 'm')
    assert out == {'category': 'contenido'}
    assert len(calls) == 2
    assert calls[0]['json_mode'] is True
    assert calls[0]['images'] is None
    # The second attempt ends with the language-correction prompt.
    last = calls[1]['messages'][-1]
    assert last['role'] == 'user'
    assert 'ISO 639-1' in last['content']
    assert 'category' in last['content']


def test_list_models_delegates_to_provider(monkeypatch):
    monkeypatch.setattr('app.ai.providers.llamacpp.list_models', lambda: ['a', 'b'])
    assert llm.list_models() == ['a', 'b']
