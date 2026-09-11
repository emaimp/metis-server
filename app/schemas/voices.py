from __future__ import annotations

from pydantic import BaseModel


class VoiceProfileMeta(BaseModel):
    """Metadata of a named reference voice (audio/image bytes served separately)."""
    id: str
    name: str
    filename: str
    content_type: str
    size: int
    duration_seconds: float | None = None
    ref_text: str | None = None
    image_filename: str | None = None
    image_content_type: str | None = None
    image_size: int | None = None
    has_image: bool = False
    created_at: str
    updated_at: str


class UpdateVoiceProfileRequest(BaseModel):
    """Patch a voice profile. Omitted fields are kept; empty ref_text clears it."""
    name: str | None = None
    ref_text: str | None = None
