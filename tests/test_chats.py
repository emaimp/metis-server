import pathlib
import sys

# Ensure the project root (containing the 'app' package) is importable
# when pytest is run from anywhere.
ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest

from app.core.database import DB_PATH, db_conn, init_db
from app.core.time_utils import now_iso
from app.repositories import chats as repo



@pytest.fixture()
def test_db(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    test_db = tmp_path / 'test_chats.db'
    monkeypatch.setattr('app.core.database.DB_PATH', test_db)
    init_db()
    return test_db



def test_create_and_get_chat(test_db):
    chat = repo.create_chat_db(None)
    assert chat['title'] == 'New chat'
    got = repo.get_chat_db(chat['id'])
    assert got is not None
    assert got['id'] == chat['id']
    assert got['messages'] == []



def test_create_chat_title_capped_and_stripped(test_db):
    chat = repo.create_chat_db('   ' + 'x' * 300 + '  ')
    assert chat['title'] == 'x' * 200



def test_list_chats_orders_and_counts(test_db):
    a = repo.create_chat_db('A')
    b = repo.create_chat_db('B')
    ts = now_iso()
    repo.add_messages_db(a['id'], 'hola', 'adiós', 'mi-modelo', 10, ts, ts)
    chats = repo.list_chats_db()
    assert [c['id'] for c in chats] == [a['id'], b['id']]
    by_id = {c['id']: c for c in chats}
    assert by_id[a['id']]['message_count'] == 2



def test_chat_exists(test_db):
    chat = repo.create_chat_db('A')
    assert repo.chat_exists_db(chat['id'])
    assert not repo.chat_exists_db('no-existe')



def test_update_title_truncates_and_strips(test_db):
    chat = repo.create_chat_db(None)
    updated = repo.update_title_db(chat['id'], '   ' + 'y' * 250 + '  ')
    assert updated is not None
    assert updated['title'] == 'y' * 200
    assert repo.update_title_db('no-existe', 'x') is None



def test_add_messages_sets_auto_title_and_order(test_db):
    chat = repo.create_chat_db(None)
    user_ts = '2026-01-01T00:00:00.000000Z'
    assistant_ts = '2026-01-01T00:00:00.000001Z'
    u, a = repo.add_messages_db(chat['id'], 'Primera pregunta del usuario', 'respuesta', 'llama3', 123, user_ts, assistant_ts)
    assert a['model'] == 'llama3'
    assert u['role'] == 'user' and u['response_time_ms'] is None
    got = repo.get_chat_db(chat['id'])
    assert got is not None
    assert len(got['messages']) == 2
    assert got['title'] == 'Primera pregunta del usuario'
    assert got['messages'][0]['role'] == 'user'
    assert got['messages'][1]['role'] == 'assistant'
    assert got['messages'][1]['response_time_ms'] == 123



def test_add_messages_missing_chat_raises(test_db):
    with pytest.raises(ValueError):
        repo.add_messages_db('no-existe', 'hi', 'hi', None, 0, now_iso(), now_iso())



def test_delete_chat_cascades_messages(test_db):
    chat = repo.create_chat_db('A')
    repo.add_messages_db(chat['id'], 'hi', 'bye', None, 0, now_iso(), now_iso())
    assert repo.delete_chat_db(chat['id'])
    assert not repo.delete_chat_db(chat['id'])
    assert repo.get_chat_db(chat['id']) is None
    with db_conn() as conn:
        cur = conn.execute('SELECT COUNT(*) FROM messages WHERE chat_id = ?', (chat['id'],))
        assert cur.fetchone()[0] == 0


def test_add_messages_with_attachments(test_db):
    chat = repo.create_chat_db(None)
    user_ts = '2026-01-01T00:00:00.000000Z'
    assistant_ts = '2026-01-01T00:00:00.000001Z'
    attachments = [
        {'filename': 'doc.txt', 'content_type': 'text/plain', 'data': b'hola mundo'},
        {'filename': 'img.png', 'content_type': 'image/png', 'data': b'\x89PNGfake'},
    ]
    u, a = repo.add_messages_db(chat['id'], 'hi', 'res', None, 0, user_ts, assistant_ts, attachments)
    assert len(u['attachments']) == 2
    assert u['attachments'][0]['filename'] == 'doc.txt'
    assert a['attachments'] == []
    got = repo.get_chat_db(chat['id'])
    assert got is not None
    user_msg = got['messages'][0]
    assert user_msg['attachments'][0]['id'] == u['attachments'][0]['id']
    assert user_msg['attachments'][1]['content_type'] == 'image/png'
    att = repo.get_attachment_db(chat['id'], u['attachments'][0]['id'])
    assert att is not None
    assert att['data'] == b'hola mundo'
    assert att['size'] == 10


def test_get_attachment_wrong_chat_returns_none(test_db):
    c1 = repo.create_chat_db('A')
    c2 = repo.create_chat_db('B')
    u, _ = repo.add_messages_db(
        c1['id'], 'hi', 'res', None, 0, now_iso(), now_iso(),
        [{'filename': 'a.txt', 'content_type': 'text/plain', 'data': b'x'}],
    )
    att_id = u['attachments'][0]['id']
    assert repo.get_attachment_db(c2['id'], att_id) is None


def test_delete_chat_cascades_attachments(test_db):
    chat = repo.create_chat_db('A')
    u, _ = repo.add_messages_db(
        chat['id'], 'hi', 'res', None, 0, now_iso(), now_iso(),
        [{'filename': 'a.txt', 'content_type': 'text/plain', 'data': b'x'}],
    )
    att_id = u['attachments'][0]['id']
    assert repo.get_attachment_db(chat['id'], att_id) is not None
    repo.delete_chat_db(chat['id'])
    assert repo.get_attachment_db(chat['id'], att_id) is None


def test_add_audio_and_get(test_db):
    chat = repo.create_chat_db(None)
    u, a = repo.add_messages_db(chat['id'], 'hi', 'res', None, 0, now_iso(), now_iso())
    wav = b'RIFFfake-wav-bytes'
    meta = repo.add_audio_db(chat['id'], a['id'], wav, 24000)
    assert meta['mime_type'] == 'audio/wav'
    assert meta['size'] == len(wav)
    got = repo.get_audio_db(chat['id'], meta['id'])
    assert got is not None
    assert got['data'] == wav
    assert got['sample_rate'] == 24000
    detail = repo.get_chat_db(chat['id'])
    assert detail['messages'][1]['audio']['id'] == meta['id']
    assert detail['messages'][0]['audio'] is None


def test_add_audio_replaces_existing(test_db):
    chat = repo.create_chat_db(None)
    u, a = repo.add_messages_db(chat['id'], 'hi', 'res', None, 0, now_iso(), now_iso())
    m1 = repo.add_audio_db(chat['id'], a['id'], b'first', 24000)
    m2 = repo.add_audio_db(chat['id'], a['id'], b'second-wav', 24000)
    assert m2['id'] != m1['id']
    assert repo.get_audio_db(chat['id'], m1['id']) is None
    assert repo.get_audio_db(chat['id'], m2['id'])['data'] == b'second-wav'


def test_add_audio_missing_message_raises(test_db):
    chat = repo.create_chat_db(None)
    with pytest.raises(ValueError):
        repo.add_audio_db(chat['id'], 'no-existe', b'x', 24000)


def test_get_message_db(test_db):
    chat = repo.create_chat_db(None)
    u, a = repo.add_messages_db(chat['id'], 'hi', 'res', None, 0, now_iso(), now_iso())
    got = repo.get_message_db(chat['id'], a['id'])
    assert got is not None
    assert got['role'] == 'assistant'
    assert repo.get_message_db(chat['id'], 'no-existe') is None


def test_delete_chat_cascades_audios(test_db):
    chat = repo.create_chat_db(None)
    u, a = repo.add_messages_db(chat['id'], 'hi', 'res', None, 0, now_iso(), now_iso())
    meta = repo.add_audio_db(chat['id'], a['id'], b'wav', 24000)
    repo.delete_chat_db(chat['id'])
    assert repo.get_audio_db(chat['id'], meta['id']) is None
