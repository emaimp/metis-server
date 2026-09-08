import pathlib
import sys

# Ensure the project root (containing the 'app' package) is importable
# when pytest is run from anywhere.
ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.database import init_db
from app.routers.chats import router


def _fake_ask_chat(message, image=None, available_tools=None, tool_hint=None, model=None):
    return "Respuesta simulada del modelo"


def _fake_resolve_model(selected: str | None = None) -> str:
    return "modelo-de-prueba"


@pytest.fixture()
def client(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    test_db = tmp_path / 'test_chats.db'
    monkeypatch.setattr('app.core.database.DB_PATH', test_db)
    init_db()
    monkeypatch.setattr('app.routers.chats.ask_chat', _fake_ask_chat)
    monkeypatch.setattr('app.routers.chats.resolve_model', _fake_resolve_model)

    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c


def _create_chat(client: TestClient, title: str | None = None) -> dict:
    payload = {} if title is None else {'title': title}
    resp = client.post('/chats', json=payload)
    assert resp.status_code == 200
    return resp.json()


def test_create_chat(client):
    chat = _create_chat(client, 'Mi chat')
    assert chat['title'] == 'Mi chat'
    assert chat['id']
    assert chat['created_at']
    assert chat['updated_at']


def test_create_chat_default_title(client):
    chat = _create_chat(client)
    assert chat['title'] == 'New chat'


def test_list_chats(client):
    a = _create_chat(client, 'A')
    b = _create_chat(client, 'B')
    chats = client.get('/chats').json()
    ids = [c['id'] for c in chats]
    assert a['id'] in ids
    assert b['id'] in ids


def test_get_chat(client):
    chat = _create_chat(client, 'A')
    resp = client.get(f"/chats/{chat['id']}")
    assert resp.status_code == 200
    data = resp.json()
    assert data['id'] == chat['id']
    assert data['messages'] == []


def test_get_chat_not_found(client):
    resp = client.get('/chats/no-existe')
    assert resp.status_code == 404


def test_update_chat_title(client):
    chat = _create_chat(client, 'A')
    resp = client.patch(f"/chats/{chat['id']}", json={'title': 'Nuevo'})
    assert resp.status_code == 200
    assert resp.json()['title'] == 'Nuevo'


def test_update_chat_title_empty(client):
    chat = _create_chat(client, 'A')
    resp = client.patch(f"/chats/{chat['id']}", json={'title': '   '})
    assert resp.status_code == 400


def test_update_chat_title_not_found(client):
    resp = client.patch('/chats/no-existe', json={'title': 'X'})
    assert resp.status_code == 404


def test_delete_chat(client):
    chat = _create_chat(client, 'A')
    resp = client.delete(f"/chats/{chat['id']}")
    assert resp.status_code == 200
    assert resp.json() == {'detail': 'Chat deleted'}
    assert client.get(f"/chats/{chat['id']}").status_code == 404


def test_delete_chat_not_found(client):
    resp = client.delete('/chats/no-existe')
    assert resp.status_code == 404


def test_create_message(client):
    chat = _create_chat(client)
    resp = client.post(f"/chats/{chat['id']}/messages", data={'message': 'Hola'})
    assert resp.status_code == 200
    data = resp.json()
    assert data['user_message']['content'] == 'Hola'
    assert data['user_message']['role'] == 'user'
    assert data['assistant_message']['content'] == 'Respuesta simulada del modelo'
    assert data['assistant_message']['model'] == 'modelo-de-prueba'
    assert len(data['chat']['messages']) == 2


def test_create_message_empty(client):
    chat = _create_chat(client)
    resp = client.post(f"/chats/{chat['id']}/messages", data={'message': '   '})
    assert resp.status_code == 400


def test_create_message_chat_not_found(client):
    resp = client.post('/chats/no-existe/messages', data={'message': 'Hola'})
    assert resp.status_code == 404


def test_create_message_resolve_model_error(client, monkeypatch):
    def _boom(selected=None):
        raise RuntimeError('No model configured')

    monkeypatch.setattr('app.routers.chats.resolve_model', _boom)
    chat = _create_chat(client)
    resp = client.post(f"/chats/{chat['id']}/messages", data={'message': 'Hola'})
    assert resp.status_code == 500


def test_create_message_ask_chat_error(client, monkeypatch):
    def _boom(message, image=None, available_tools=None, tool_hint=None, model=None):
        raise ConnectionError('ollama down')

    monkeypatch.setattr('app.routers.chats.ask_chat', _boom)
    chat = _create_chat(client)
    resp = client.post(f"/chats/{chat['id']}/messages", data={'message': 'Hola'})
    assert resp.status_code == 502


def test_create_message_empty_response(client, monkeypatch):
    def _empty(message, image=None, available_tools=None, tool_hint=None, model=None):
        return ''

    monkeypatch.setattr('app.routers.chats.ask_chat', _empty)
    chat = _create_chat(client)
    resp = client.post(f"/chats/{chat['id']}/messages", data={'message': 'Hola'})
    assert resp.status_code == 502


def test_create_message_with_document(client):
    chat = _create_chat(client)
    content = b'contenido de prueba del documento'
    resp = client.post(
        f"/chats/{chat['id']}/messages",
        data={'message': 'Analiza esto'},
        files={'document': ('nota.txt', content, 'text/plain')},
    )
    assert resp.status_code == 200
    data = resp.json()
    user_msg = data['user_message']
    assert len(user_msg['attachments']) == 1
    att = user_msg['attachments'][0]
    assert att['filename'] == 'nota.txt'
    assert att['content_type'] == 'text/plain'
    assert att['size'] == len(content)
    assert data['chat']['messages'][0]['attachments'][0]['id'] == att['id']


def test_create_message_with_image(client):
    chat = _create_chat(client)
    resp = client.post(
        f"/chats/{chat['id']}/messages",
        data={'message': 'Que hay en la imagen'},
        files={'image': ('foto.png', b'fake-png-bytes', 'image/png')},
    )
    assert resp.status_code == 200
    data = resp.json()
    att = data['user_message']['attachments'][0]
    assert att['filename'] == 'foto.png'
    assert att['content_type'] == 'image/png'
    assert att['size'] == len(b'fake-png-bytes')


def test_download_attachment(client):
    chat = _create_chat(client)
    resp = client.post(
        f"/chats/{chat['id']}/messages",
        data={'message': 'Que hay en la imagen'},
        files={'image': ('foto.png', b'fake-png-bytes', 'image/png')},
    )
    assert resp.status_code == 200
    att = resp.json()['user_message']['attachments'][0]
    dl = client.get(f"/chats/{chat['id']}/attachments/{att['id']}")
    assert dl.status_code == 200
    assert dl.content == b'fake-png-bytes'
    assert dl.headers['content-type'] == 'image/png'


def test_download_attachment_wrong_chat(client):
    chat_a = _create_chat(client, 'A')
    chat_b = _create_chat(client, 'B')
    resp = client.post(
        f"/chats/{chat_a['id']}/messages",
        data={'message': 'Que hay en la imagen'},
        files={'image': ('foto.png', b'fake-png-bytes', 'image/png')},
    )
    att = resp.json()['user_message']['attachments'][0]
    dl = client.get(f"/chats/{chat_b['id']}/attachments/{att['id']}")
    assert dl.status_code == 404


def test_download_attachment_not_found(client):
    chat = _create_chat(client)
    resp = client.get(f"/chats/{chat['id']}/attachments/no-existe")
    assert resp.status_code == 404


def test_create_message_unsupported_document(client):
    chat = _create_chat(client)
    resp = client.post(
        f"/chats/{chat['id']}/messages",
        data={'message': 'x'},
        files={'document': ('malo.exe', b'x', 'application/octet-stream')},
    )
    assert resp.status_code == 400


def test_create_message_image_too_large(client, monkeypatch):
    monkeypatch.setattr('app.core.uploads.MAX_UPLOAD_BYTES', 3)
    chat = _create_chat(client)
    resp = client.post(
        f"/chats/{chat['id']}/messages",
        data={'message': 'x'},
        files={'image': ('foto.png', b'large-bytes', 'image/png')},
    )
    assert resp.status_code == 400
