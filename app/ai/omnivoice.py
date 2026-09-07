from __future__ import annotations

import tempfile
import threading
from pathlib import Path

import torch

SAMPLE_RATE = 24000
DEFAULT_REF_AUDIO: Path | None = None
DEFAULT_REF_TEXT: str | None = None
DEFAULT_INSTRUCT: str = ""

_model = None
_lock = threading.Lock()


def _get_model():
    # Lazy, thread-safe singleton for OmniVoice.
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        from omnivoice import OmniVoice

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # float16 only viable on CUDA; fallback to float32 on CPU.
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        _model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice",
            device_map=device,
            dtype=dtype,
        )
        return _model


def _resolve_ref_audio(ref_audio_path: str | None) -> str | None:
    if ref_audio_path:
        p = Path(ref_audio_path)
        if not p.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")
        return str(p)
    if isinstance(DEFAULT_REF_AUDIO, Path) and DEFAULT_REF_AUDIO.exists():
        return str(DEFAULT_REF_AUDIO)
    return None


def _resolve_instruct(instruct: str | None) -> str | None:
    effective = instruct.strip() if instruct and instruct.strip() else None
    if effective is None and DEFAULT_INSTRUCT and DEFAULT_INSTRUCT.strip():
        effective = DEFAULT_INSTRUCT.strip()
    if effective:
        from app.ai.tts.voices import validate_instruct

        validate_instruct(effective)
        return effective
    return None


def synthesize(
    text: str,
    ref_audio_path: str | None = None,
    ref_text: str | None = None,
    instruct: str | None = None,
):
    # Generate raw audio ndarray at 24kHz.
    if not text or not text.strip():
        raise ValueError("text must be non-empty")
    model = _get_model()
    ref_audio = _resolve_ref_audio(ref_audio_path)
    instruct_val = _resolve_instruct(instruct)

    # Auto-transcribe if ref_text not provided (omit ref_text -> Whisper).
    effective_ref_text = ref_text.strip() if ref_text and ref_text.strip() else None
    if effective_ref_text is None and DEFAULT_REF_TEXT and DEFAULT_REF_TEXT.strip():
        effective_ref_text = DEFAULT_REF_TEXT.strip()

    kwargs: dict = {"text": text}
    if ref_audio is not None:
        kwargs["ref_audio"] = ref_audio
    if effective_ref_text is not None:
        kwargs["ref_text"] = effective_ref_text
    if instruct_val is not None:
        kwargs["instruct"] = instruct_val

    audio = model.generate(**kwargs)
    # audio is list[np.ndarray]
    return audio[0], SAMPLE_RATE


def synthesize_with_ref_bytes(
    text: str,
    ref_audio_bytes: bytes,
    ref_audio_suffix: str = ".wav",
    ref_text: str | None = None,
    instruct: str | None = None,
):
    # Generate audio using reference audio provided as bytes.
    if not text or not text.strip():
        raise ValueError("text must be non-empty")
    if not ref_audio_bytes:
        raise ValueError("ref_audio_bytes must be non-empty")
    # OmniVoice expects a file path, so write to temp file.
    suffix = ref_audio_suffix if ref_audio_suffix.startswith(".") else f".{ref_audio_suffix}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(ref_audio_bytes)
        temp_path = f.name
    try:
        model = _get_model()
        instruct_val = _resolve_instruct(instruct)

        effective_ref_text = ref_text.strip() if ref_text and ref_text.strip() else None
        if effective_ref_text is None and DEFAULT_REF_TEXT and DEFAULT_REF_TEXT.strip():
            effective_ref_text = DEFAULT_REF_TEXT.strip()

        kwargs: dict = {"text": text, "ref_audio": temp_path}
        if effective_ref_text is not None:
            kwargs["ref_text"] = effective_ref_text
        if instruct_val is not None:
            kwargs["instruct"] = instruct_val
        audio = model.generate(**kwargs)
        return audio[0], SAMPLE_RATE
    finally:
        Path(temp_path).unlink(missing_ok=True)
