from __future__ import annotations

import base64
import io

import soundfile as sf


def ndarray_to_wav_bytes(audio, sample_rate: int) -> bytes:
    # Convert numpy ndarray audio to WAV bytes.
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    return buf.getvalue()


def wav_bytes_to_base64(wav_bytes: bytes) -> str:
    # Encode WAV bytes to base64 string.
    return base64.b64encode(wav_bytes).decode("utf-8")


def base64_to_wav_bytes(b64: str) -> bytes:
    # Decode base64 string to WAV bytes.
    return base64.b64decode(b64)


def ndarray_to_base64(audio, sample_rate: int) -> str:
    # Convert ndarray audio directly to base64 WAV.
    return wav_bytes_to_base64(ndarray_to_wav_bytes(audio, sample_rate))

