from __future__ import annotations

import sqlite3
import uuid

from app.core.database import db_conn
from app.core.time_utils import now_iso

MAX_PROFILE_NAME_LENGTH = 100


class DuplicateVoiceProfileError(ValueError):
    """Raised when a voice profile name is already taken (case-insensitive)."""


# Sentinel so PATCH can distinguish "field absent" (keep) from explicit values.
UNSET: object = object()


def _new_id() -> str:
    return uuid.uuid4().hex


def normalize_profile_name(name: str | None) -> str:
    """Strip and validate a voice profile name; raise ValueError on empty/oversize."""
    normalized = (name or "").strip()
    if not normalized:
        raise ValueError("Voice profile name must be non-empty")
    if len(normalized) > MAX_PROFILE_NAME_LENGTH:
        raise ValueError(f"Voice profile name must be at most {MAX_PROFILE_NAME_LENGTH} chars")
    return normalized


def normalize_ref_text(ref_text: str | None) -> str | None:
    """Trim an optional reference transcription; empty becomes None."""
    if ref_text is None:
        return None
    stripped = ref_text.strip()
    return stripped or None


def _meta_from_full(full: dict) -> dict:
    meta = {k: v for k, v in full.items() if k not in ("data", "image_data")}
    meta["has_image"] = full.get("image_data") is not None
    return meta


def create_voice_profile_db(
    name: str,
    filename: str,
    content_type: str,
    data: bytes,
    duration_seconds: float | None = None,
    ref_text: str | None = None,
    image_filename: str | None = None,
    image_content_type: str | None = None,
    image_data: bytes | None = None,
) -> dict:
    """Persist a named reference voice (global library) and return its metadata (no BLOBs)."""
    normalized = normalize_profile_name(name)
    if not data:
        raise ValueError("Reference audio must be non-empty")
    now = now_iso()
    profile_id = _new_id()
    image_size = len(image_data) if image_data else None
    if image_data and (not image_filename or not image_content_type):
        raise ValueError("Image filename and content type are required with image data")
    trimmed_ref_text = normalize_ref_text(ref_text)
    with db_conn() as conn:
        try:
            conn.execute(
                """INSERT INTO voice_profiles
                (id, name, filename, content_type, size, duration_seconds, data, ref_text,
                 image_filename, image_content_type, image_size, image_data, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    profile_id, normalized, filename, content_type, len(data),
                    duration_seconds, data, trimmed_ref_text,
                    image_filename, image_content_type, image_size, image_data, now, now,
                ),
            )
        except sqlite3.IntegrityError:
            raise DuplicateVoiceProfileError(f"Voice profile name already exists: '{normalized}'")
    return {
        "id": profile_id, "name": normalized, "filename": filename,
        "content_type": content_type, "size": len(data),
        "duration_seconds": duration_seconds, "ref_text": trimmed_ref_text,
        "image_filename": image_filename, "image_content_type": image_content_type,
        "image_size": image_size, "has_image": image_data is not None,
        "created_at": now, "updated_at": now,
    }


def list_voice_profiles_db() -> list[dict]:
    """List voice profiles (metadata only, no BLOBs), ordered by name."""
    with db_conn() as conn:
        cur = conn.execute(
            """SELECT id, name, filename, content_type, size, duration_seconds, ref_text,
            image_filename, image_content_type, image_size,
            CASE WHEN image_data IS NOT NULL THEN 1 ELSE 0 END AS has_image,
            created_at, updated_at
            FROM voice_profiles ORDER BY name COLLATE NOCASE ASC"""
        )
        rows = cur.fetchall()
        return [
            {
                "id": r["id"], "name": r["name"], "filename": r["filename"],
                "content_type": r["content_type"], "size": r["size"],
                "duration_seconds": r["duration_seconds"], "ref_text": r["ref_text"],
                "image_filename": r["image_filename"],
                "image_content_type": r["image_content_type"], "image_size": r["image_size"],
                "has_image": bool(r["has_image"]),
                "created_at": r["created_at"], "updated_at": r["updated_at"],
            }
            for r in rows
        ]


def get_voice_profile_db(profile_id: str) -> dict | None:
    """Return a voice profile with its BLOBs, or None if it does not exist."""
    with db_conn() as conn:
        cur = conn.execute("SELECT * FROM voice_profiles WHERE id = ?", (profile_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def get_voice_profile_meta_db(profile_id: str) -> dict | None:
    """Return a voice profile's metadata (no BLOBs), or None if missing."""
    full = get_voice_profile_db(profile_id)
    return _meta_from_full(full) if full else None


def update_voice_profile_db(
    profile_id: str, name: str | None = None, ref_text: str | None | object = UNSET,
) -> dict | None:
    """Rename a profile and/or set its ref_text.

    - name=None: keep; otherwise normalized (ValueError on empty/oversize).
    - ref_text=UNSET (default) or None: keep; empty string clears to NULL; otherwise trimmed.
    Returns updated metadata (no BLOBs), or None if the profile does not exist.
    """
    with db_conn() as conn:
        cur = conn.execute("SELECT id FROM voice_profiles WHERE id = ?", (profile_id,))
        if not cur.fetchone():
            return None
        updates: list[str] = []
        params: list = []
        if name is not None:
            normalized = normalize_profile_name(name)
            updates.append("name = ?")
            params.append(normalized)
        if ref_text is not UNSET and ref_text is not None:
            updates.append("ref_text = ?")
            params.append(normalize_ref_text(ref_text))  # type: ignore[arg-type]
        if not updates:
            full = get_voice_profile_db(profile_id)
            return _meta_from_full(full) if full else None
        updates.append("updated_at = ?")
        params.append(now_iso())
        params.append(profile_id)
        try:
            conn.execute(
                f"UPDATE voice_profiles SET {', '.join(updates)} WHERE id = ?",
                params,
            )
        except sqlite3.IntegrityError:
            raise DuplicateVoiceProfileError("Voice profile name already exists")
    full = get_voice_profile_db(profile_id)
    return _meta_from_full(full) if full else None


def set_voice_profile_image_db(
    profile_id: str, image_filename: str, image_content_type: str, image_data: bytes,
) -> dict | None:
    """Replace (or set) a profile's image; return metadata or None if missing."""
    if not image_data:
        raise ValueError("Image data must be non-empty")
    now = now_iso()
    with db_conn() as conn:
        cur = conn.execute(
            """UPDATE voice_profiles SET image_filename = ?, image_content_type = ?,
            image_size = ?, image_data = ?, updated_at = ? WHERE id = ?""",
            (image_filename, image_content_type, len(image_data), image_data, now, profile_id),
        )
        if cur.rowcount == 0:
            return None
    full = get_voice_profile_db(profile_id)
    return _meta_from_full(full) if full else None


def clear_voice_profile_image_db(profile_id: str) -> bool:
    """Remove a profile's image; True if the image was removed, False otherwise."""
    now = now_iso()
    with db_conn() as conn:
        cur = conn.execute(
            """UPDATE voice_profiles SET image_filename = NULL, image_content_type = NULL,
            image_size = NULL, image_data = NULL, updated_at = ?
            WHERE id = ? AND image_data IS NOT NULL""",
            (now, profile_id),
        )
        return cur.rowcount > 0


def delete_voice_profile_db(profile_id: str) -> bool:
    """Delete a voice profile; audios already generated with it are kept."""
    with db_conn() as conn:
        cur = conn.execute("DELETE FROM voice_profiles WHERE id = ?", (profile_id,))
        return cur.rowcount > 0
