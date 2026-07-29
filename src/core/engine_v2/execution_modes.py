"""Pure certified V2 execution-mode and combination validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


class ExecutionModeValidationError(ValueError):
    def __init__(self, message: str, *, field: str = "execution") -> None:
        self.field = field
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class PositionModeState:
    target_mode: str
    trail_mode: str
    trail_activation_mode: str
    max_days_enabled: bool
    boundary_mode: str
    margin_mode: str
    price_rounding_mode: str


@dataclass(frozen=True, slots=True)
class SignalReversalModeState:
    stop_mode: str
    boundary_mode: str


_POSITION_FIELDS = frozenset(
    {
        "entryOrder",
        "stop",
        "target",
        "trail",
        "trailActivation",
        "sizing",
        "maxDays",
        "boundary",
        "margin",
        "priceRounding",
    }
)
_SIGNAL_FIELDS = frozenset(
    {
        "topology",
        "entryOrder",
        "stop",
        "target",
        "trail",
        "trailActivation",
        "sizing",
        "exitOnSignal",
        "maxDays",
        "boundary",
        "margin",
        "priceRounding",
    }
)


def _require(modes: Mapping[str, str], field: str, expected: str, family: str) -> None:
    actual = modes.get(field)
    if actual != expected:
        raise ExecutionModeValidationError(
            f"Unsupported {family} execution mode {field}={actual!r}; expected {expected!r}.",
            field=field,
        )


def _reject_unknown(modes: Mapping[str, str], allowed: frozenset[str], family: str) -> None:
    unknown = sorted(set(modes) - set(allowed))
    if unknown:
        raise ExecutionModeValidationError(
            f"Unsupported {family} execution field(s): {unknown}.",
            field=unknown[0],
        )


def resolve_position_mode_state(modes: Mapping[str, str]) -> PositionModeState:
    """Validate and resolve the certified position/bracket family."""

    _reject_unknown(modes, _POSITION_FIELDS, "position/bracket")
    _require(modes, "entryOrder", "market_next_open", "position/bracket")
    _require(modes, "stop", "atr_swing", "position/bracket")
    _require(modes, "sizing", "risk_per_trade", "position/bracket")

    target_mode = modes.get("target")
    trail_mode = modes.get("trail")
    if target_mode not in {"rr", "none"}:
        raise ExecutionModeValidationError(
            f"Unsupported position/bracket execution mode target={target_mode!r}; expected 'rr' or 'none'.",
            field="target",
        )
    if trail_mode not in {"ma", "none"}:
        raise ExecutionModeValidationError(
            f"Unsupported position/bracket execution mode trail={trail_mode!r}; expected 'ma' or 'none'.",
            field="trail",
        )
    trail_activation_mode = modes.get("trailActivation", "none")
    if trail_activation_mode not in {"none", "rr"}:
        raise ExecutionModeValidationError(
            "Unsupported position/bracket execution mode "
            f"trailActivation={trail_activation_mode!r}; expected 'none' or 'rr'.",
            field="trailActivation",
        )
    valid_target = target_mode == "rr" and trail_mode == "none" and trail_activation_mode == "none"
    valid_trail = target_mode == "none" and trail_mode == "ma" and trail_activation_mode == "rr"
    if not (valid_target or valid_trail):
        raise ExecutionModeValidationError(
            "Position/bracket supports target=rr with trail=none/trailActivation=none "
            "or target=none with trail=ma/trailActivation=rr.",
            field="trailActivation",
        )

    max_days = modes.get("maxDays", "false")
    if max_days not in {"true", "false"}:
        raise ExecutionModeValidationError(
            f"Unsupported position/bracket execution mode maxDays={max_days!r}; expected 'true' or 'false'.",
            field="maxDays",
        )
    boundary = modes.get("boundary", "strict_close")
    if boundary not in {"strict_close", "none"}:
        raise ExecutionModeValidationError(
            f"Unsupported position/bracket execution mode boundary={boundary!r}.",
            field="boundary",
        )
    margin = modes.get("margin", "off")
    if margin not in {"off", "report_only"}:
        raise ExecutionModeValidationError(
            f"Unsupported position/bracket execution mode margin={margin!r}.",
            field="margin",
        )
    rounding = modes.get("priceRounding", "none")
    if rounding not in {"none", "tick_outward"}:
        raise ExecutionModeValidationError(
            f"Unsupported position/bracket execution mode priceRounding={rounding!r}.",
            field="priceRounding",
        )
    return PositionModeState(
        target_mode=target_mode,
        trail_mode=trail_mode,
        trail_activation_mode=trail_activation_mode,
        max_days_enabled=max_days == "true",
        boundary_mode=boundary,
        margin_mode=margin,
        price_rounding_mode=rounding,
    )


def resolve_signal_reversal_mode_state(modes: Mapping[str, str]) -> SignalReversalModeState:
    """Validate and resolve the certified signal-reversal family."""

    _reject_unknown(modes, _SIGNAL_FIELDS, "signal_reversal")
    _require(modes, "topology", "signal_reversal", "signal_reversal")
    _require(modes, "entryOrder", "market_next_open", "signal_reversal")
    _require(modes, "sizing", "fixed_pct_equity", "signal_reversal")
    _require(modes, "exitOnSignal", "true", "signal_reversal")
    _require(modes, "priceRounding", "none", "signal_reversal")

    stop_mode = modes.get("stop")
    if stop_mode not in {"none", "emergency_pct"}:
        raise ExecutionModeValidationError(
            f"Unsupported signal_reversal execution mode stop={stop_mode!r}; expected 'none' or 'emergency_pct'.",
            field="stop",
        )
    for field, expected in (
        ("target", "none"),
        ("trail", "none"),
        ("trailActivation", "none"),
        ("maxDays", "false"),
        ("margin", "off"),
    ):
        actual = modes.get(field)
        if actual is not None and actual != expected:
            raise ExecutionModeValidationError(
                f"Unsupported signal_reversal execution mode {field}={actual!r}; expected {expected!r} or absent.",
                field=field,
            )
    boundary = modes.get("boundary", "strict_close")
    if boundary not in {"strict_close", "none"}:
        raise ExecutionModeValidationError(
            f"Unsupported signal_reversal execution mode boundary={boundary!r}.",
            field="boundary",
        )
    return SignalReversalModeState(stop_mode=stop_mode, boundary_mode=boundary)


def validate_execution_modes(modes: Mapping[str, str]) -> str:
    """Validate one resolved variant and return its certified family name."""

    topology = modes.get("topology")
    if topology == "signal_reversal":
        resolve_signal_reversal_mode_state(modes)
        return "signal_reversal"
    if topology is None:
        resolve_position_mode_state(modes)
        return "position"
    raise ExecutionModeValidationError(
        f"Unsupported V2 execution topology: {topology!r}.",
        field="topology",
    )


__all__ = [
    "ExecutionModeValidationError",
    "PositionModeState",
    "SignalReversalModeState",
    "resolve_position_mode_state",
    "resolve_signal_reversal_mode_state",
    "validate_execution_modes",
]
