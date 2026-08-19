"""Side-effect-free canonical JSON serialization for Strategy Lab analysis."""

from __future__ import annotations

import json
from typing import Any

from .dataset import AnalysisError


def canonical_json_bytes(value: Any) -> bytes:
    """Return the frozen canonical JSON representation as UTF-8 bytes."""
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"canonical JSON: {exc}") from None
    return payload.encode("utf-8")
