from __future__ import annotations

import datetime


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string ending in 'Z'."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
