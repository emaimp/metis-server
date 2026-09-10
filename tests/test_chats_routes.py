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


def _fake_ask_chat(message, image=None, model=None):
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
    def _boom(message, image=None, model=None):
        raise ConnectionError('ollama down')

    monkeypatch.setattr('app.routers.chats.ask_chat', _boom)
    chat = _create_chat(client)
    resp = client.post(f"/chats/{chat['id']}/messages", data={'message': 'Hola'})
    assert resp.status_code == 502


def test_create_message_empty_response(client, monkeypatch):
    def _empty(message, image=None, model=None):
        return ''

    monkeypatch.setattr('app.routers.chats.ask_chat', _empty)
    chat = _create_chat(client)
    resp = client.post(f"/chats/{chat['id']}/messages", data={'message': 'Hola'})
    assert resp.status_code == 502


def test_create_message_with_document_embeds_text(client, monkeypatch):
    captured = {}

    def _fake_ask_chat(message, image=None, model=None):
        captured['message'] = message
        return 'Respuesta simulada del modelo'

    monkeypatch.setattr('app.routers.chats.ask_chat', _fake_ask_chat)
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
    # The document text is embedded server-side in the message sent to the model
    assert 'contenido de prueba del documento' in captured['message']
    assert 'Analiza esto' in captured['message']
    assert 'nota.txt' in captured['message']


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


def test_create_message_with_tts(client, monkeypatch):
    calls = []

    def _fake_synthesize(text, ref_audio_path=None, ref_text=None, instruct=None):
        import numpy as np

        calls.append({'text': text, 'instruct': instruct})
        return np.zeros(2400, dtype=np.float32), 24000

    monkeypatch.setattr('app.ai.omnivoice.synthesize', _fake_synthesize)
    chat = _create_chat(client)
    resp = client.post(
        f"/chats/{chat['id']}/messages",
        data={'message': 'Hola', 'tts': 'true'},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert calls and calls[0]['text'] == 'Respuesta simulada del modelo'
    audio = data['assistant_message']['audio']
    assert audio is not None
    assert audio['mime_type'] == 'audio/wav'
    assert audio['sample_rate'] == 24000
    assert data['mime_type'] == 'audio/wav'
    assert data['sample_rate'] == 24000
    import base64

    assert base64.b64decode(data['audio_base64']).startswith(b'RIFF')
    # Stored audio is downloadable
    dl = client.get(f"/chats/{chat['id']}/audios/{audio['id']}")
    assert dl.status_code == 200
    assert dl.headers['content-type'] == 'audio/wav'
    assert dl.content.startswith(b'RIFF')


def test_create_message_without_tts_has_no_audio(client):
    chat = _create_chat(client)
    resp = client.post(f"/chats/{chat['id']}/messages", data={'message': 'Hola'})
    assert resp.status_code == 200
    data = resp.json()
    assert data['assistant_message']['audio'] is None
    assert data['audio_base64'] is None


def test_create_message_tts_no_speakable(client, monkeypatch):
    monkeypatch.setattr('app.routers.chats.strip_markdown', lambda text: '')
    chat = _create_chat(client)
    resp = client.post(f"/chats/{chat['id']}/messages", data={'message': 'Hola', 'tts': 'true'})
    assert resp.status_code == 502
    assert 'no speakable' in resp.json()['detail']


def test_create_message_tts_invalid_voice(client, monkeypatch):
    def _fake_synthesize(*args, **kwargs):
        raise ValueError('Invalid instruct attribute(s): male, female')

    monkeypatch.setattr('app.ai.omnivoice.synthesize', _fake_synthesize)
    chat = _create_chat(client)
    resp = client.post(
        f"/chats/{chat['id']}/messages",
        data={'message': 'Hola', 'tts': 'true', 'voice': 'male, female'},
    )
    assert resp.status_code == 400


def test_create_message_tts_failure_keeps_messages(client, monkeypatch):
    def _fake_synthesize(*args, **kwargs):
        raise RuntimeError('CUDA out of memory')

    monkeypatch.setattr('app.ai.omnivoice.synthesize', _fake_synthesize)
    chat = _create_chat(client)
    resp = client.post(f"/chats/{chat['id']}/messages", data={'message': 'Hola', 'tts': 'true'})
    assert resp.status_code == 502
    assert resp.json()['detail'].startswith('TTS generation failed')
    # The exchange remains persisted; the audio can be regenerated on demand
    detail = client.get(f"/chats/{chat['id']}").json()
    assert len(detail['messages']) == 2
    assert detail['messages'][1]['audio'] is None


def test_generate_audio_on_demand(client, monkeypatch):
    calls = []

    def _fake_synthesize(text, ref_audio_path=None, ref_text=None, instruct=None):
        import numpy as np

        calls.append(text)
        return np.zeros(2400, dtype=np.float32), 24000

    monkeypatch.setattr('app.ai.omnivoice.synthesize', _fake_synthesize)
    chat = _create_chat(client)
    resp = client.post(f"/chats/{chat['id']}/messages", data={'message': 'Hola'})
    msg_id = resp.json()['assistant_message']['id']
    first = client.post(f"/chats/{chat['id']}/messages/{msg_id}/audio")
    assert first.status_code == 200
    audio_id = first.json()['audio']['id']
    assert first.json()['audio_base64']
    # Linked to the assistant message
    detail = client.get(f"/chats/{chat['id']}").json()
    assert detail['messages'][1]['audio']['id'] == audio_id
    # Idempotent: second call returns the stored audio without regenerating
    second = client.post(f"/chats/{chat['id']}/messages/{msg_id}/audio")
    assert second.status_code == 200
    assert second.json()['audio']['id'] == audio_id
    assert len(calls) == 1


def test_generate_audio_on_demand_user_message_rejected(client):
    chat = _create_chat(client)
    resp = client.post(f"/chats/{chat['id']}/messages", data={'message': 'Hola'})
    user_id = resp.json()['user_message']['id']
    r = client.post(f"/chats/{chat['id']}/messages/{user_id}/audio")
    assert r.status_code == 400


def test_generate_audio_missing_message(client):
    chat = _create_chat(client)
    r = client.post(f"/chats/{chat['id']}/messages/no-existe/audio")
    assert r.status_code == 404


def test_download_audio_not_found(client):
    chat = _create_chat(client)
    r = client.get(f"/chats/{chat['id']}/audios/no-existe")
    assert r.status_code == 404


def test_create_message_rename_tool(client, monkeypatch):
    captured = {}

    def _fake_rename_file(data, extension, existing_files=None, instruction='', model=None):
        captured['data'] = data
        captured['extension'] = extension
        captured['instruction'] = instruction
        captured['existing_files'] = existing_files
        return 'factura_2026'

    monkeypatch.setattr('app.routers.chats.rename_file', _fake_rename_file)
    chat = _create_chat(client)
    resp = client.post(
        f"/chats/{chat['id']}/messages",
        data={
            'message': '@tool_rename usa formato fecha_name',
            'existing_files': 'otro.pdf,doc2.pdf',
        },
        files={'document': ('contrato.pdf', b'pdf-bytes', 'application/pdf')},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data['new_name'] == 'factura_2026'
    assert captured['instruction'] == 'usa formato fecha_name'
    assert captured['existing_files'] == ['otro.pdf', 'doc2.pdf']
    # The exchange is persisted: user message (as typed) + assistant message (result)
    assert data['user_message']['content'] == '@tool_rename usa formato fecha_name'
    assert data['assistant_message']['content'] == 'factura_2026'
    assert data['user_message']['attachments'][0]['filename'] == 'contrato.pdf'
    detail = client.get(f"/chats/{chat['id']}").json()
    assert len(detail['messages']) == 2
    assert detail['messages'][0]['attachments'][0]['id'] == data['user_message']['attachments'][0]['id']


def test_create_message_categorize_tool(client, monkeypatch):
    def _fake_categorize(data, extension, existing_categories=None, instruction='', model=None):
        assert instruction == 'es una factura'
        assert existing_categories == ['contratos', 'facturas']
        return 'facturas'

    monkeypatch.setattr('app.routers.chats.categorize_file', _fake_categorize)
    chat = _create_chat(client)
    resp = client.post(
        f"/chats/{chat['id']}/messages",
        data={'message': '@tool_categorize es una factura', 'existing_categories': 'contratos,facturas'},
        files={'document': ('doc.pdf', b'pdf-bytes', 'application/pdf')},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data['category'] == 'facturas'
    assert data['assistant_message']['content'] == 'facturas'


def test_create_message_rename_unassigned_persists(client, monkeypatch):
    monkeypatch.setattr('app.routers.chats.rename_file', lambda *args, **kwargs: 'unassigned')
    chat = _create_chat(client)
    resp = client.post(
        f"/chats/{chat['id']}/messages",
        data={'message': '@tool_rename'},
        files={'document': ('doc.txt', b'x', 'text/plain')},
    )
    assert resp.status_code == 200
    assert resp.json()['new_name'] == 'unassigned'
    detail = client.get(f"/chats/{chat['id']}").json()
    assert len(detail['messages']) == 2


def test_create_message_rename_error_compensates(client, monkeypatch):
    monkeypatch.setattr('app.routers.chats.rename_file', lambda *args, **kwargs: 'Error: unreadable document')
    chat = _create_chat(client)
    resp = client.post(
        f"/chats/{chat['id']}/messages",
        data={'message': '@tool_rename'},
        files={'document': ('doc.txt', b'x', 'text/plain')},
    )
    assert resp.status_code == 502
    # Nothing persisted: the chat has no messages (attachment compensated away)
    detail = client.get(f"/chats/{chat['id']}").json()
    assert detail['messages'] == []


def test_create_message_unknown_tool(client):
    chat = _create_chat(client)
    resp = client.post(
        f"/chats/{chat['id']}/messages",
        data={'message': '@tool_translate hola'},
        files={'document': ('doc.txt', b'x', 'text/plain')},
    )
    assert resp.status_code == 400
    assert "Unknown tool '@tool_translate'" in resp.json()['detail']


def test_create_message_mention_without_attachment(client):
    chat = _create_chat(client)
    resp = client.post(f"/chats/{chat['id']}/messages", data={'message': '@tool_rename hola'})
    assert resp.status_code == 400
    assert resp.json()['detail'] == 'Tool mentions require an attached document or image'
