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

_POSITION_MODE_VALUES = {
    "entryOrder": frozenset({"market_next_open"}),
    "stop": frozenset({"atr_swing"}),
    "target": frozenset({"rr", "none"}),
    "trail": frozenset({"ma", "none"}),
    "trailActivation": frozenset({"none", "rr"}),
    "sizing": frozenset({"risk_per_trade"}),
    "maxDays": frozenset({"true", "false"}),
    "boundary": frozenset({"strict_close", "none"}),
    "margin": frozenset({"off", "report_only"}),
    "priceRounding": frozenset({"none", "tick_outward"}),
}
_SIGNAL_MODE_VALUES = {
    "topology": frozenset({"signal_reversal"}),
    "entryOrder": frozenset({"market_next_open"}),
    "stop": frozenset({"none", "emergency_pct"}),
    "target": frozenset({"none"}),
    "trail": frozenset({"none"}),
    "trailActivation": frozenset({"none"}),
    "sizing": frozenset({"fixed_pct_equity"}),
    "exitOnSignal": frozenset({"true"}),
    "maxDays": frozenset({"false"}),
    "boundary": frozenset({"strict_close", "none"}),
    "margin": frozenset({"off"}),
    "priceRounding": frozenset({"none"}),
}
_MODE_VALUES_BY_FAMILY = {
    "position": _POSITION_MODE_VALUES,
    "signal_reversal": _SIGNAL_MODE_VALUES,
}


def execution_family_supports_mode_value(
    family: str,
    mode_field: str,
    mode_value: str,
) -> bool:
    """Return whether a canonical mode value is supported by one V2 family.

    Coupled combination validity remains the responsibility of the family
    resolver; this authority answers whether the individual value participates
    in any valid combination for that family.
    """

    family_values = _MODE_VALUES_BY_FAMILY.get(str(family))
    if family_values is None:
        return False
    return str(mode_value) in family_values.get(str(mode_field), ())


def _require(modes: Mapping[str, str], field: str, expected: str, family: str) -> None:
    actual = modes.get(field)
    if actual != expected:
        raise ExecutionModeValidationError(
            f"Unsupported {family} execution mode {field}={actual!r}; expected {expected!r}.",
            field=field,
        )


def _only(values: frozenset[str]) -> str:
    """Return the sole supported value for a required single-value field."""

    expected, = values
    return expected


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
    _require(modes, "entryOrder", _only(_POSITION_MODE_VALUES["entryOrder"]), "position/bracket")
    _require(modes, "stop", _only(_POSITION_MODE_VALUES["stop"]), "position/bracket")
    _require(modes, "sizing", _only(_POSITION_MODE_VALUES["sizing"]), "position/bracket")

    target_mode = modes.get("target")
    trail_mode = modes.get("trail")
    if target_mode not in _POSITION_MODE_VALUES["target"]:
        raise ExecutionModeValidationError(
            f"Unsupported position/bracket execution mode target={target_mode!r}; expected 'rr' or 'none'.",
            field="target",
        )
    if trail_mode not in _POSITION_MODE_VALUES["trail"]:
        raise ExecutionModeValidationError(
            f"Unsupported position/bracket execution mode trail={trail_mode!r}; expected 'ma' or 'none'.",
            field="trail",
        )
    trail_activation_mode = modes.get("trailActivation", "none")
    if trail_activation_mode not in _POSITION_MODE_VALUES["trailActivation"]:
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
    if max_days not in _POSITION_MODE_VALUES["maxDays"]:
        raise ExecutionModeValidationError(
            f"Unsupported position/bracket execution mode maxDays={max_days!r}; expected 'true' or 'false'.",
            field="maxDays",
        )
    boundary = modes.get("boundary", "strict_close")
    if boundary not in _POSITION_MODE_VALUES["boundary"]:
        raise ExecutionModeValidationError(
            f"Unsupported position/bracket execution mode boundary={boundary!r}.",
            field="boundary",
        )
    margin = modes.get("margin", "off")
    if margin not in _POSITION_MODE_VALUES["margin"]:
        raise ExecutionModeValidationError(
            f"Unsupported position/bracket execution mode margin={margin!r}.",
            field="margin",
        )
    rounding = modes.get("priceRounding", "none")
    if rounding not in _POSITION_MODE_VALUES["priceRounding"]:
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
    for field in ("topology", "entryOrder", "sizing", "exitOnSignal", "priceRounding"):
        _require(modes, field, _only(_SIGNAL_MODE_VALUES[field]), "signal_reversal")

    stop_mode = modes.get("stop")
    if stop_mode not in _SIGNAL_MODE_VALUES["stop"]:
        raise ExecutionModeValidationError(
            f"Unsupported signal_reversal execution mode stop={stop_mode!r}; expected 'none' or 'emergency_pct'.",
            field="stop",
        )
    for field in ("target", "trail", "trailActivation", "maxDays", "margin"):
        expected = _only(_SIGNAL_MODE_VALUES[field])
        actual = modes.get(field)
        if actual is not None and actual != expected:
            raise ExecutionModeValidationError(
                f"Unsupported signal_reversal execution mode {field}={actual!r}; expected {expected!r} or absent.",
                field=field,
            )
    boundary = modes.get("boundary", "strict_close")
    if boundary not in _SIGNAL_MODE_VALUES["boundary"]:
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
    "execution_family_supports_mode_value",
    "resolve_position_mode_state",
    "resolve_signal_reversal_mode_state",
    "validate_execution_modes",
]
