import json
import pathlib
import sys

# Ensure the project root (containing the 'app' package) is importable
# when pytest is run from anywhere.
ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.core.database import init_db
from app.routers.chats import router


def _fake_stream(message, model=None, image=None):
    yield 'Hola '
    yield 'mundo'


def _fake_resolve_model(selected: str | None = None) -> str:
    return 'modelo-de-prueba'


@pytest.fixture()
def client(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    test_db = tmp_path / 'test_chats_stream.db'
    monkeypatch.setattr('app.core.database.DB_PATH', test_db)
    init_db()
    monkeypatch.setattr('app.routers.chats.ask_chat_stream', _fake_stream)
    monkeypatch.setattr('app.routers.chats.resolve_model', _fake_resolve_model)

    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c


def _create_chat(client: TestClient) -> dict:
    resp = client.post('/chats', json={})
    assert resp.status_code == 200
    return resp.json()


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.split('\n\n'):
        block = block.strip()
        if not block:
            continue
        event = None
        data = None
        for line in block.split('\n'):
            if line.startswith('event: '):
                event = line[7:].strip()
            elif line.startswith('data: '):
                data = line[6:].strip()
        events.append((event, json.loads(data) if data is not None else None))
    return events


def test_message_stream_deltas_and_done(client):
    chat = _create_chat(client)
    resp = client.post(
        f"/chats/{chat['id']}/messages/stream",
        data={'message': 'Hola', 'model': 'm'},
    )
    assert resp.status_code == 200
    assert resp.headers['content-type'] == 'text/event-stream; charset=utf-8'
    events = _parse_sse(resp.text)
    assert [e for e, _ in events] == ['delta', 'delta', 'done']
    assert events[0][1] == {'text': 'Hola '}
    assert events[1][1] == {'text': 'mundo'}
    done = events[2][1]
    assert done['user_message']['content'] == 'Hola'
    assert done['assistant_message']['content'] == 'Hola mundo'
    assert done['assistant_message']['model'] == 'modelo-de-prueba'
    assert done['audio_base64'] is None
    assert done['mime_type'] is None
    assert done['sample_rate'] is None
    assert done['tts_error'] is None
    # The exchange is persisted exactly like the blocking endpoint
    detail = client.get(f"/chats/{chat['id']}").json()
    assert [m['content'] for m in detail['messages']] == ['Hola', 'Hola mundo']


def test_message_stream_with_image_links_attachment(client):
    chat = _create_chat(client)
    resp = client.post(
        f"/chats/{chat['id']}/messages/stream",
        data={'message': 'Mira', 'model': 'm'},
        files={'image': ('foto.png', b'fake-image', 'image/png')},
    )
    events = _parse_sse(resp.text)
    done = events[-1][1]
    assert done['user_message']['attachments'][0]['filename'] == 'foto.png'
    att_id = done['user_message']['attachments'][0]['id']
    assert client.get(f"/chats/{chat['id']}/attachments/{att_id}").status_code == 200


def test_message_stream_with_document_embeds_text(client, monkeypatch):
    captured = {}

    def _fake_stream(message, model=None, image=None):
        captured['message'] = message
        yield 'listo'

    monkeypatch.setattr('app.routers.chats.ask_chat_stream', _fake_stream)
    chat = _create_chat(client)
    resp = client.post(
        f"/chats/{chat['id']}/messages/stream",
        data={'message': 'Analiza esto', 'model': 'm'},
        files={'document': ('doc.txt', b'contenido de prueba del documento', 'text/plain')},
    )
    events = _parse_sse(resp.text)
    assert events[-1][0] == 'done'
    assert 'contenido de prueba del documento' in captured['message']
    assert 'Analiza esto' in captured['message']


def test_message_stream_error_event_and_compensation(client, monkeypatch):
    def _boom(message, model=None, image=None):
        raise ConnectionError('llm down')
        yield  # pragma: no cover - makes this a generator

    monkeypatch.setattr('app.routers.chats.ask_chat_stream', _boom)
    chat = _create_chat(client)
    resp = client.post(
        f"/chats/{chat['id']}/messages/stream",
        data={'message': 'Hola', 'model': 'm'},
        files={'image': ('foto.png', b'fake-image', 'image/png')},
    )
    assert _parse_sse(resp.text) == [('error', {'detail': 'Error querying the model: llm down'})]
    # Nothing persisted and the pending attachment was compensated (deleted)
    assert client.get(f"/chats/{chat['id']}").json()['messages'] == []


def test_message_stream_empty_response(client, monkeypatch):
    def _empty(message, model=None, image=None):
        return
        yield  # pragma: no cover - makes this a generator

    monkeypatch.setattr('app.routers.chats.ask_chat_stream', _empty)
    chat = _create_chat(client)
    resp = client.post(
        f"/chats/{chat['id']}/messages/stream",
        data={'message': 'Hola', 'model': 'm'},
    )
    assert _parse_sse(resp.text) == [('error', {'detail': 'Empty model response'})]
    assert client.get(f"/chats/{chat['id']}").json()['messages'] == []


def test_message_stream_tts_rides_in_done_event(client, monkeypatch):
    async def _fake_tts(chat_id, message_id, content, voice, voice_profile_id=None):
        return {
            'audio': {'id': 'a1', 'mime_type': 'audio/wav', 'sample_rate': 24000, 'size': 10, 'created_at': 't'},
            'audio_base64': 'QUJD',
            'mime_type': 'audio/wav',
            'sample_rate': 24000,
        }

    monkeypatch.setattr('app.routers.chats._synthesize_and_store', _fake_tts)
    chat = _create_chat(client)
    resp = client.post(
        f"/chats/{chat['id']}/messages/stream",
        data={'message': 'Hola', 'model': 'm', 'tts': 'true'},
    )
    events = _parse_sse(resp.text)
    assert [e for e, _ in events] == ['delta', 'delta', 'done']
    done = events[-1][1]
    assert done['audio_base64'] == 'QUJD'
    assert done['mime_type'] == 'audio/wav'
    assert done['sample_rate'] == 24000
    assert done['assistant_message']['audio']['id'] == 'a1'
    assert done['tts_error'] is None


def test_message_stream_tts_failure_keeps_done_with_error(client, monkeypatch):
    async def _boom_tts(chat_id, message_id, content, voice, voice_profile_id=None):
        raise HTTPException(status_code=502, detail='TTS generation failed: boom')

    monkeypatch.setattr('app.routers.chats._synthesize_and_store', _boom_tts)
    chat = _create_chat(client)
    resp = client.post(
        f"/chats/{chat['id']}/messages/stream",
        data={'message': 'Hola', 'model': 'm', 'tts': 'true'},
    )
    done = _parse_sse(resp.text)[-1]
    assert done[0] == 'done'
    assert done[1]['audio_base64'] is None
    assert done[1]['tts_error'] == 'TTS generation failed: boom'
    # The streamed text is persisted anyway
    detail = client.get(f"/chats/{chat['id']}").json()
    assert [m['content'] for m in detail['messages']] == ['Hola', 'Hola mundo']


def test_message_stream_tool_mention_rejected(client):
    chat = _create_chat(client)
    resp = client.post(
        f"/chats/{chat['id']}/messages/stream",
        data={'message': '@tool_rename dale', 'model': 'm'},
    )
    assert resp.status_code == 400
    assert 'streaming' in resp.json()['detail']


def test_message_stream_requires_model(client, monkeypatch):
    def _boom(selected=None):
        raise RuntimeError('No model selected')

    monkeypatch.setattr('app.routers.chats.resolve_model', _boom)
    chat = _create_chat(client)
    resp = client.post(
        f"/chats/{chat['id']}/messages/stream",
        data={'message': 'Hola'},
    )
    assert resp.status_code == 500


def test_message_stream_chat_not_found(client):
    resp = client.post('/chats/no-existe/messages/stream', data={'message': 'Hola', 'model': 'm'})
    assert resp.status_code == 404


def test_message_stream_empty_message_rejected(client):
    chat = _create_chat(client)
    resp = client.post(
        f"/chats/{chat['id']}/messages/stream",
        data={'message': '   ', 'model': 'm'},
    )
    assert resp.status_code == 400
