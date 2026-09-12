import io
import pathlib
import sys

# Ensure the project root (containing the 'app' package) is importable
# when pytest is run from anywhere.
ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.database import init_db
from app.routers.chats import router as chats_router
from app.routers.voices import router as voices_router
from app.repositories import voice_profiles as vp_repo


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _wav_bytes(seconds: float = 5.0, sample_rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(seconds * sample_rate), dtype="float32"), sample_rate, format="WAV")
    return buf.getvalue()


def _mp3_bytes(seconds: float = 5.0, sample_rate: int = 24000) -> bytes:
    # libsndfile 1.1+ encodes MP3 natively, so tests need no ffmpeg/encoder.
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(seconds * sample_rate), dtype="float32"), sample_rate, format="MP3")
    return buf.getvalue()


@pytest.fixture()
def db(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr('app.core.database.DB_PATH', tmp_path / 'test_voices.db')
    init_db()


@pytest.fixture()
def client(db):
    app = FastAPI()
    app.include_router(voices_router)
    app.include_router(chats_router)
    with TestClient(app) as c:
        yield c


def _create_profile(
    client: TestClient,
    name: str = 'Marta',
    file_bytes: bytes | None = None,
    filename: str = 'ref.wav',
    mime: str = 'audio/wav',
    image_bytes: bytes | None = None,
    image_name: str = 'avatar.png',
    image_mime: str = 'image/png',
    **extra,
) -> dict:
    data = {'name': name, **extra}
    files = {'file': (filename, file_bytes if file_bytes is not None else _wav_bytes(), mime)}
    if image_bytes is not None:
        files['image'] = (image_name, image_bytes, image_mime)
    resp = client.post('/voices/profiles', data=data, files=files)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _create_chat(client: TestClient) -> dict:
    resp = client.post('/chats', json={})
    assert resp.status_code == 200
    return resp.json()


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


def test_repo_create_get_list_delete(db):
    meta = vp_repo.create_voice_profile_db(
        'Voz Uno', 'ref.wav', 'audio/wav', _wav_bytes(4), duration_seconds=4.0, ref_text='  hola  ',
    )
    assert meta['id'] and meta['name'] == 'Voz Uno'
    assert meta['ref_text'] == 'hola'  # trimmed
    assert meta['size'] > 0 and meta['duration_seconds'] == 4.0
    assert meta['has_image'] is False and meta['image_size'] is None

    full = vp_repo.get_voice_profile_db(meta['id'])
    assert full['data'] and full['ref_text'] == 'hola'

    # Metadata (no BLOBs) matches the create response
    assert vp_repo.get_voice_profile_meta_db(meta['id']) == meta

    listed = vp_repo.list_voice_profiles_db()
    assert [p['id'] for p in listed] == [meta['id']]
    assert 'data' not in listed[0] and 'image_data' not in listed[0]

    assert vp_repo.delete_voice_profile_db(meta['id']) is True
    assert vp_repo.delete_voice_profile_db(meta['id']) is False
    assert vp_repo.get_voice_profile_db(meta['id']) is None


def test_repo_duplicate_name_case_insensitive(db):
    vp_repo.create_voice_profile_db('Voz', 'a.wav', 'audio/wav', _wav_bytes())
    with pytest.raises(vp_repo.DuplicateVoiceProfileError):
        vp_repo.create_voice_profile_db('  voz  ', 'b.wav', 'audio/wav', _wav_bytes())


def test_repo_rename_and_image(db):
    meta = vp_repo.create_voice_profile_db('Original', 'a.wav', 'audio/wav', _wav_bytes(), ref_text='hola')
    pid = meta['id']
    # Rename; omitted ref_text (UNSET) is kept
    updated = vp_repo.update_voice_profile_db(pid, name='  Renombrada  ')
    assert updated['name'] == 'Renombrada'
    assert updated['ref_text'] == 'hola'

    # ref_text semantics: "" / whitespace clears to NULL, non-empty is trimmed
    assert vp_repo.update_voice_profile_db(pid, ref_text='')['ref_text'] is None
    assert vp_repo.update_voice_profile_db(pid, ref_text='   ')['ref_text'] is None
    assert vp_repo.update_voice_profile_db(pid, ref_text='hola mundo')['ref_text'] == 'hola mundo'

    # Renaming onto an existing (case-insensitive) name -> duplicate
    vp_repo.update_voice_profile_db(pid, name='Otra Vez')
    other = vp_repo.create_voice_profile_db('Conflicto', 'b.wav', 'audio/wav', _wav_bytes())
    with pytest.raises(vp_repo.DuplicateVoiceProfileError):
        vp_repo.update_voice_profile_db(other['id'], name='OTRA VEZ')

    # Image set / clear
    img = vp_repo.set_voice_profile_image_db(pid, 'avatar.png', 'image/png', b'png-bytes')
    assert img['has_image'] is True and img['image_size'] == len(b'png-bytes')
    assert vp_repo.set_voice_profile_image_db('no-existe', 'a.png', 'image/png', b'x') is None
    assert vp_repo.clear_voice_profile_image_db(pid) is True
    assert vp_repo.clear_voice_profile_image_db(pid) is False
    assert vp_repo.update_voice_profile_db(pid, name='Final')['has_image'] is False


# ---------------------------------------------------------------------------
# POST /voices/profiles — validations and passthrough
# ---------------------------------------------------------------------------


def test_create_profile_validations(client):
    wav = _wav_bytes()
    # Empty / oversized name
    r = client.post('/voices/profiles', data={'name': '   '}, files={'file': ('ref.wav', wav, 'audio/wav')})
    assert r.status_code == 400
    r = client.post('/voices/profiles', data={'name': 'x' * 101}, files={'file': ('ref.wav', wav, 'audio/wav')})
    assert r.status_code == 400
    # Unsupported extension (only .wav/.mp3 allowed)
    r = client.post('/voices/profiles', data={'name': 'Ok'}, files={'file': ('ref.ogg', wav, 'audio/ogg')})
    assert r.status_code == 400
    assert '.mp3' in r.json()['detail'] and '.wav' in r.json()['detail']
    # Oversized upload (> 10 MB)
    r = client.post(
        '/voices/profiles',
        data={'name': 'Ok'},
        files={'file': ('ref.wav', b'\\0' * (10 * 1024 * 1024 + 1), 'audio/wav')},
    )
    assert r.status_code == 400
    # Unparseable audio (valid extension, garbage bytes)
    r = client.post('/voices/profiles', data={'name': 'Ok'}, files={'file': ('ref.wav', b'not-audio', 'audio/wav')})
    assert r.status_code == 400
    assert 'Invalid audio' in r.json()['detail']
    # Empty audio bytes
    r = client.post('/voices/profiles', data={'name': 'Ok'}, files={'file': ('ref.wav', b'', 'audio/wav')})
    assert r.status_code == 400
    # Duplicate name (case-insensitive) -> 409
    _create_profile(client, name='Marta')
    r = client.post('/voices/profiles', data={'name': 'marta'}, files={'file': ('ref.wav', wav, 'audio/wav')})
    assert r.status_code == 409


def test_reference_duration_helper():
    from app.routers.voices import _reference_duration_seconds

    assert abs(_reference_duration_seconds(_wav_bytes(5.0)) - 5.0) < 0.05
    assert abs(_reference_duration_seconds(_mp3_bytes(5.0)) - 5.0) < 0.1
    with pytest.raises(ValueError):
        _reference_duration_seconds(b'not-audio')


def test_create_profile_mp3_passthrough(client):
    mp3 = _mp3_bytes(5.0)
    meta = _create_profile(client, name='Mp3 Voice', file_bytes=mp3, filename='voz.mp3', mime='audio/mpeg')
    assert meta['content_type'] == 'audio/mpeg'
    assert meta['size'] == len(mp3)
    assert abs(meta['duration_seconds'] - 5.0) < 0.1
    # Stored as-is: the download returns the exact uploaded bytes with the real MIME
    r = client.get(f"/voices/profiles/{meta['id']}/audio")
    assert r.status_code == 200
    assert r.content == mp3
    assert r.headers['content-type'].startswith('audio/mpeg')
    assert 'filename*=UTF-8' in r.headers['content-disposition']
    listed = client.get('/voices/profiles').json()
    assert listed[0]['content_type'] == 'audio/mpeg' and listed[0]['size'] == len(mp3)
    assert listed[0]['duration_seconds'] > 0


def test_create_profile_with_mp3(client):
    # MP3 goes through the same route as WAV (native format, no transcoding)
    meta = _create_profile(
        client, name='Con Mp3', file_bytes=_mp3_bytes(4.0), filename='ref.mp3', mime='audio/mpeg',
        ref_text='dice hola',
    )
    assert meta['ref_text'] == 'dice hola'
    full = client.get(f"/voices/profiles/{meta['id']}").json()
    assert full['filename'] == 'ref.mp3' and full['duration_seconds'] > 0


def test_create_profile_short_audio_warns(client):
    meta = _create_profile(client, name='Corta', file_bytes=_wav_bytes(1.0))
    assert 'warning' in meta
    assert '3-10s' in meta['warning']
    # In-range duration has no warning
    meta_ok = _create_profile(client, name='En Rango')
    assert 'warning' not in meta_ok


def test_create_profile_with_image_and_downloads(client):
    meta = _create_profile(client, name='Con Imagen', image_bytes=b'png-bytes')
    assert meta['has_image'] is True
    assert meta['image_filename'] == 'avatar.png'
    assert meta['image_content_type'] == 'image/png'
    assert meta['image_size'] == len(b'png-bytes')
    pid = meta['id']
    r = client.get(f'/voices/profiles/{pid}/image')
    assert r.status_code == 200 and r.content == b'png-bytes'
    # Metadata-only endpoints keep has_image (no bytes in JSON)
    assert client.get(f'/voices/profiles/{pid}').json()['has_image'] is True
    # A profile without image -> image endpoint 404
    other = _create_profile(client, name='Sin Imagen')
    assert client.get(f"/voices/profiles/{other['id']}/image").status_code == 404
    # Audio download works; unknown ids -> 404 everywhere
    assert client.get(f'/voices/profiles/{pid}/audio').status_code == 200
    assert client.get('/voices/profiles/no-existe').status_code == 404
    assert client.get('/voices/profiles/no-existe/audio').status_code == 404
    assert client.get('/voices/profiles/no-existe/image').status_code == 404


# ---------------------------------------------------------------------------
# PATCH / PUT image / DELETE
# ---------------------------------------------------------------------------


def test_patch_put_delete_image_and_profile(client):
    meta = _create_profile(client, name='Original', image_bytes=b'png-1')
    pid = meta['id']
    # PATCH name (omitted ref_text kept)
    r = client.patch(f'/voices/profiles/{pid}', json={'name': 'Renombrada'})
    assert r.status_code == 200 and r.json()['name'] == 'Renombrada'
    # PATCH ref_text: "" clears, "hola" sets
    assert client.patch(f'/voices/profiles/{pid}', json={'ref_text': 'hola'}).json()['ref_text'] == 'hola'
    assert client.patch(f'/voices/profiles/{pid}', json={'ref_text': ''}).json()['ref_text'] is None
    # PATCH nothing -> 400; empty name -> 400
    assert client.patch(f'/voices/profiles/{pid}', json={}).status_code == 400
    assert client.patch(f'/voices/profiles/{pid}', json={'name': '  '}).status_code == 400
    # PATCH duplicate name -> 409; unknown profile -> 404
    _create_profile(client, name='Otra')
    assert client.patch(f'/voices/profiles/{pid}', json={'name': 'otra'}).status_code == 409
    assert client.patch('/voices/profiles/no-existe', json={'name': 'X'}).status_code == 404
    # PUT image replaces
    r = client.put(f'/voices/profiles/{pid}/image', files={'image': ('logo.png', b'png-2', 'image/png')})
    assert r.status_code == 200 and r.json()['image_filename'] == 'logo.png'
    assert client.get(f'/voices/profiles/{pid}/image').content == b'png-2'
    # Unsupported image extension -> 400
    r = client.put(f'/voices/profiles/{pid}/image', files={'image': ('logo.svg', b'x', 'image/svg+xml')})
    assert r.status_code == 400
    # DELETE image
    assert client.delete(f'/voices/profiles/{pid}/image').json() == {'detail': 'Image deleted'}
    assert client.get(f'/voices/profiles/{pid}/image').status_code == 404
    assert client.delete(f'/voices/profiles/{pid}/image').status_code == 404
    # DELETE profile
    assert client.delete(f'/voices/profiles/{pid}').json() == {'detail': 'Voice profile deleted'}
    assert client.get(f'/voices/profiles/{pid}').status_code == 404
    assert client.delete(f'/voices/profiles/{pid}').status_code == 404


# ---------------------------------------------------------------------------
# TTS with a cloned voice (voice_profile_id)
# ---------------------------------------------------------------------------


def _patch_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        'app.routers.chats.ask_chat',
        lambda message, model=None, image=None: 'Respuesta simulada del modelo',
    )
    monkeypatch.setattr('app.routers.chats.resolve_model', lambda selected=None: 'modelo-de-prueba')


def _fake_synthesize_with_ref(calls: list):
    def _fake(text, ref_audio_bytes, ref_audio_suffix='.wav', ref_text=None, instruct=None):
        calls.append({'suffix': ref_audio_suffix, 'ref_text': ref_text, 'instruct': instruct})
        return np.zeros(2400, dtype=np.float32), 24000

    return _fake


def test_tts_voice_and_profile_rejected(client, monkeypatch):
    _patch_model(monkeypatch)
    chat = _create_chat(client)
    r = client.post(
        f"/chats/{chat['id']}/messages",
        data={'message': 'Hola', 'tts': 'true', 'voice': 'male', 'voice_profile_id': 'x'},
    )
    assert r.status_code == 400
    assert "not both" in r.json()['detail']


def test_tts_unknown_profile_404(client, monkeypatch):
    _patch_model(monkeypatch)
    chat = _create_chat(client)
    r = client.post(
        f"/chats/{chat['id']}/messages",
        data={'message': 'Hola', 'tts': 'true', 'voice_profile_id': 'no-existe'},
    )
    assert r.status_code == 404
    assert r.json()['detail'] == 'Voice profile not found'


def test_tts_with_voice_profile(client, monkeypatch):
    calls: list = []
    _patch_model(monkeypatch)
    monkeypatch.setattr('app.ai.omnivoice.synthesize_with_ref_bytes', _fake_synthesize_with_ref(calls))
    meta = _create_profile(client, name='Marta', ref_text='dice hola')
    chat = _create_chat(client)
    r = client.post(
        f"/chats/{chat['id']}/messages",
        data={'message': 'Hola', 'tts': 'true', 'voice_profile_id': meta['id']},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    # The stored reference is forwarded with its format suffix and transcription
    assert calls == [{'suffix': '.wav', 'ref_text': 'dice hola', 'instruct': None}]
    assert data['audio_base64'] and data['mime_type'] == 'audio/wav'
    assert data['sample_rate'] == 24000
    audio_meta = data['assistant_message']['audio']
    assert audio_meta['id'] and audio_meta['mime_type'] == 'audio/wav'
    detail = client.get(f"/chats/{chat['id']}").json()
    assert detail['messages'][1]['audio']['id'] == audio_meta['id']


def test_tts_with_mp3_profile_forwards_mp3_suffix(client, monkeypatch):
    calls: list = []
    _patch_model(monkeypatch)
    monkeypatch.setattr('app.ai.omnivoice.synthesize_with_ref_bytes', _fake_synthesize_with_ref(calls))
    meta = _create_profile(
        client, name='Mp3 Tts', file_bytes=_mp3_bytes(4.0), filename='voz.mp3', mime='audio/mpeg',
    )
    chat = _create_chat(client)
    r = client.post(
        f"/chats/{chat['id']}/messages",
        data={'message': 'Hola', 'tts': 'true', 'voice_profile_id': meta['id']},
    )
    assert r.status_code == 200, r.text
    assert calls[0]['suffix'] == '.mp3'
    assert r.json()['assistant_message']['audio']['id']


def test_on_demand_audio_with_profile_idempotent(client, monkeypatch):
    calls: list = []
    _patch_model(monkeypatch)
    monkeypatch.setattr('app.ai.omnivoice.synthesize_with_ref_bytes', _fake_synthesize_with_ref(calls))
    meta = _create_profile(client, name='On Demand')
    chat = _create_chat(client)
    r = client.post(f"/chats/{chat['id']}/messages", data={'message': 'Hola'})
    msg_id = r.json()['assistant_message']['id']
    first = client.post(
        f"/chats/{chat['id']}/messages/{msg_id}/audio", params={'voice_profile_id': meta['id']},
    )
    assert first.status_code == 200
    audio_id = first.json()['audio']['id']
    assert first.json()['audio_base64']
    assert calls and calls[0]['suffix'] == '.wav'
    # Idempotent: second call returns the stored audio without re-synthesizing
    second = client.post(
        f"/chats/{chat['id']}/messages/{msg_id}/audio", params={'voice_profile_id': meta['id']},
    )
    assert second.status_code == 200
    assert second.json()['audio']['id'] == audio_id
    assert len(calls) == 1



