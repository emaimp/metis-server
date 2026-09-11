from __future__ import annotations

import io
from pathlib import Path
from urllib.parse import quote

import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from starlette.concurrency import run_in_threadpool

from app.ai.tts.voices import VOICE_CATEGORIES
from app.core.settings import IMAGE_EXTENSIONS, VOICE_REFERENCE_EXTENSIONS
from app.core.uploads import content_type_for, validate_upload
from app.repositories.voice_profiles import (
    UNSET,
    DuplicateVoiceProfileError,
    clear_voice_profile_image_db,
    create_voice_profile_db,
    delete_voice_profile_db,
    get_voice_profile_db,
    get_voice_profile_meta_db,
    list_voice_profiles_db,
    normalize_profile_name,
    set_voice_profile_image_db,
    update_voice_profile_db,
)
from app.schemas.voices import UpdateVoiceProfileRequest, VoiceProfileMeta

router = APIRouter(tags=["voices"])


@router.get("/voices")
async def list_voices():
    # Return Voice Design categories and attributes. No presets - user selects per category.
    categories = []
    for cat_id, cat in VOICE_CATEGORIES.items():
        categories.append(
            {
                "id": cat_id,
                "label": cat["label"],
                "attributes": cat["attributes"],
            }
        )
    return {"categories": categories}


def _reference_duration_seconds(data: bytes) -> float:
    """Return the duration of reference audio bytes (.wav/.mp3), raising ValueError if unparseable.

    Mirrors OmniVoice's own loader (omnivoice/utils/audio.py): soundfile covers
    WAV (and MP3 where libsndfile ships MP3 support); librosa covers the rest.
    """
    buf = io.BytesIO(data)
    try:
        with sf.SoundFile(buf) as f:
            if not f.samplerate:
                raise ValueError("Audio has no sample rate")
            return len(f) / float(f.samplerate)
    except Exception:
        # soundfile cannot parse this format here; fall back to librosa.
        # librosa stays imported lazily: it is heavy and only needed for formats
        # soundfile cannot read (e.g. libsndfile builds without MP3 support).
        buf.seek(0)
        try:
            import librosa

            y, sr = librosa.load(buf, sr=None, mono=False)
        except Exception as e:
            raise ValueError(f"Invalid audio file: expected a parseable .wav or .mp3: {e}") from e
        if not sr or y.shape[-1] == 0:
            raise ValueError("Invalid audio file: empty or unreadable audio")
        return float(y.shape[-1]) / float(sr)


def _duration_warning(duration: float | None) -> str | None:
    if duration is None:
        return None
    if duration < 3 or duration > 10:
        return (
            f"Reference audio is {duration:.1f}s; "
            "3-10s is recommended for best cloning quality"
        )
    return None


@router.post("/voices/profiles")
async def create_voice_profile(
    name: str = Form(...),
    ref_text: str | None = Form(None),
    file: UploadFile = File(...),
    image: UploadFile | None = File(None),
):
    """Upload a named reference voice (.wav/.mp3, optional image) for voice cloning."""
    try:
        normalize_profile_name(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    data = await file.read()
    ext = validate_upload(file.filename, data, VOICE_REFERENCE_EXTENSIONS)
    try:
        duration_value: float | None = await run_in_threadpool(_reference_duration_seconds, data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    image_filename: str | None = None
    image_content_type: str | None = None
    image_data: bytes | None = None
    if image is not None:
        image_data = await image.read()
        validate_upload(image.filename, image_data, IMAGE_EXTENSIONS)
        image_filename = image.filename
        image_content_type = content_type_for(Path(image.filename or "").suffix.lower())

    try:
        meta = await run_in_threadpool(
            create_voice_profile_db,
            name, file.filename, content_type_for(ext), data, duration_value,
            ref_text, image_filename, image_content_type, image_data,
        )
    except DuplicateVoiceProfileError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    response = dict(meta)
    warning = _duration_warning(duration_value)
    if warning:
        response["warning"] = warning  # type: ignore[assignment]
    return response


@router.get("/voices/profiles", response_model=list[VoiceProfileMeta])
async def list_voice_profiles():
    """List named reference voices (metadata only, no audio/image bytes)."""
    return await run_in_threadpool(list_voice_profiles_db)


@router.get("/voices/profiles/{profile_id}", response_model=VoiceProfileMeta)
async def get_voice_profile(profile_id: str):
    """Get a voice profile's metadata, or 404."""
    meta = await run_in_threadpool(get_voice_profile_meta_db, profile_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Voice profile not found")
    return meta


@router.get("/voices/profiles/{profile_id}/audio")
async def download_voice_profile_audio(profile_id: str):
    """Download a voice profile's reference audio bytes (.wav/.mp3 as uploaded). Returns 404 if not found."""
    profile = await run_in_threadpool(get_voice_profile_db, profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Voice profile not found")
    headers = {
        "Content-Disposition": f"inline; filename*=UTF-8''{quote(profile['filename'])}",
    }
    return Response(
        content=profile["data"],
        media_type=profile["content_type"],
        headers=headers,
    )


@router.get("/voices/profiles/{profile_id}/image")
async def download_voice_profile_image(profile_id: str):
    """Download a voice profile's image. 404 if the profile or image is missing."""
    profile = await run_in_threadpool(get_voice_profile_db, profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Voice profile not found")
    if not profile.get("image_data"):
        raise HTTPException(status_code=404, detail="Image not found")
    headers = {
        "Content-Disposition": f"inline; filename*=UTF-8''{quote(profile['image_filename'])}",
    }
    return Response(
        content=profile["image_data"],
        media_type=profile["image_content_type"],
        headers=headers,
    )


@router.patch("/voices/profiles/{profile_id}", response_model=VoiceProfileMeta)
async def update_voice_profile(profile_id: str, payload: UpdateVoiceProfileRequest):
    """Rename a profile and/or set its reference transcription.

    Omitted fields are kept. Send `"ref_text": ""` to clear the transcription
    (falls back to Whisper auto-transcription at synthesis time).
    """
    provided = payload.model_fields_set
    if "name" not in provided and "ref_text" not in provided:
        raise HTTPException(status_code=400, detail="Nothing to update")
    if "name" in provided and payload.name is not None:
        try:
            normalize_profile_name(payload.name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    ref_text_value: str | None | object = UNSET
    if "ref_text" in provided:
        # "" clears to NULL via the repository normalizer; non-empty is trimmed.
        ref_text_value = payload.ref_text or ""
    try:
        meta = await run_in_threadpool(
            update_voice_profile_db,
            profile_id,
            payload.name if "name" in provided else None,
            ref_text_value,
        )
    except DuplicateVoiceProfileError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not meta:
        raise HTTPException(status_code=404, detail="Voice profile not found")
    return meta


@router.put("/voices/profiles/{profile_id}/image", response_model=VoiceProfileMeta)
async def replace_voice_profile_image(profile_id: str, image: UploadFile = File(...)):
    """Replace (or set) a voice profile's image."""
    image_data = await image.read()
    validate_upload(image.filename, image_data, IMAGE_EXTENSIONS)
    image_content_type = content_type_for(Path(image.filename or "").suffix.lower())
    meta = await run_in_threadpool(
        set_voice_profile_image_db, profile_id, image.filename, image_content_type, image_data,
    )
    if not meta:
        raise HTTPException(status_code=404, detail="Voice profile not found")
    return meta


@router.delete("/voices/profiles/{profile_id}/image")
async def delete_voice_profile_image(profile_id: str):
    """Remove a voice profile's image. 404 if the profile or image is missing."""
    profile = await run_in_threadpool(get_voice_profile_db, profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Voice profile not found")
    if not profile.get("image_data"):
        raise HTTPException(status_code=404, detail="Image not found")
    await run_in_threadpool(clear_voice_profile_image_db, profile_id)
    return {"detail": "Image deleted"}


@router.delete("/voices/profiles/{profile_id}")
async def delete_voice_profile(profile_id: str):
    """Delete a voice profile (audios already generated with it are kept), or 404."""
    deleted = await run_in_threadpool(delete_voice_profile_db, profile_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Voice profile not found")
    return {"detail": "Voice profile deleted"}
