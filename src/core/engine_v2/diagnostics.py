"""Structured Backtester V2 validation diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class V2Diagnostic:
    """Deterministic JSON-safe validation fact."""

    severity: str
    code: str
    strategy_id: str
    path: str
    variant: str | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class V2ValidationError(ValueError):
    """Base error carrying one or more structured fatal diagnostics."""

    def __init__(self, diagnostics: V2Diagnostic | Iterable[V2Diagnostic]) -> None:
        if isinstance(diagnostics, V2Diagnostic):
            normalized = (diagnostics,)
        else:
            normalized = tuple(diagnostics)
        if not normalized:
            raise ValueError("V2ValidationError requires at least one diagnostic.")
        self.diagnostics = normalized
        super().__init__("; ".join(item.message for item in normalized))

    def to_dict(self) -> dict[str, Any]:
        return {"diagnostics": [item.to_dict() for item in self.diagnostics]}


def diagnostic_from_mapping(payload: Mapping[str, Any]) -> V2Diagnostic:
    """Hydrate a diagnostic from persisted or API-compatible metadata."""

    return V2Diagnostic(
        severity=str(payload.get("severity") or "error"),
        code=str(payload.get("code") or "V2_VALIDATION_ERROR"),
        strategy_id=str(payload.get("strategy_id") or "<unknown strategy>"),
        path=str(payload.get("path") or ""),
        variant=None if payload.get("variant") is None else str(payload.get("variant")),
        message=str(payload.get("message") or "V2 validation failed."),
    )


__all__ = ["V2Diagnostic", "V2ValidationError", "diagnostic_from_mapping"]
