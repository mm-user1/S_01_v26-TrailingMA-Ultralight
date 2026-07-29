"""Reusable canonical runtime declarations for V2 config fixtures."""

from __future__ import annotations

from copy import deepcopy


_CANONICAL_DECLARATIONS = {
    "dateFilter": {
        "type": "bool",
        "default": True,
        "role": "runtime",
        "optimize": {"enabled": False},
    },
    "start": {
        "type": "datetime",
        "default": None,
        "role": "runtime",
        "optimize": {"enabled": False},
    },
    "end": {
        "type": "datetime",
        "default": None,
        "role": "runtime",
        "optimize": {"enabled": False},
    },
    "warmupBars": {
        "type": "int",
        "default": 1000,
        "min": 100,
        "max": 5000,
        "step": 50,
        "role": "runtime",
        "optimize": {"enabled": False},
    },
}


def canonical_v2_runtime_declarations() -> dict[str, dict]:
    return deepcopy(_CANONICAL_DECLARATIONS)

