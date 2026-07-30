"""Canonical core-owned Backtester V2 runtime contract."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from .diagnostics import V2Diagnostic, V2ValidationError


V2_RUNTIME_CONTRACT_VERSION = "v2_runtime_contract_v1"
V2_RESERVED_RUNTIME_PARAM_NAMES = (
    "dateFilter",
    "start",
    "end",
    "warmupBars",
)
V2_REBASABLE_DATE_PARAM_NAMES = frozenset({"dateFilter", "start", "end"})

_TRUE_VALUES = frozenset({"true", "1", "yes", "y", "on"})
_FALSE_VALUES = frozenset({"false", "0", "no", "n", "off"})


@dataclass(frozen=True, slots=True)
class V2RuntimeField:
    name: str
    type: str
    ui_default: Any
    legacy_default: Any
    minimum: int | None = None
    maximum: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


V2_RUNTIME_FIELDS = (
    V2RuntimeField("dateFilter", "bool", True, False),
    V2RuntimeField("start", "datetime", None, None),
    V2RuntimeField("end", "datetime", None, None),
    V2RuntimeField("warmupBars", "int", 1000, 1000, 100, 5000),
)
_RUNTIME_FIELD_BY_NAME = {field.name: field for field in V2_RUNTIME_FIELDS}


class V2RuntimeValidationError(V2ValidationError):
    """Raised when V2 runtime values or declarations violate the contract."""


def runtime_contract_payload() -> dict[str, Any]:
    """Return the ordered JSON-safe runtime contract for later API delivery."""

    return {
        "version": V2_RUNTIME_CONTRACT_VERSION,
        "fields": [field.to_dict() for field in V2_RUNTIME_FIELDS],
    }


def _runtime_error(
    *,
    strategy_id: str,
    path: str,
    message: str,
    code: str = "V2_INVALID_RUNTIME_VALUE",
) -> V2RuntimeValidationError:
    return V2RuntimeValidationError(
        V2Diagnostic(
            severity="error",
            code=code,
            strategy_id=strategy_id,
            path=path,
            variant=None,
            message=message,
        )
    )


def _strict_bool(value: Any, *, strategy_id: str, path: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(float(value)) and float(value) in {0.0, 1.0}:
            return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
    raise _runtime_error(
        strategy_id=strategy_id,
        path=path,
        message=f"{strategy_id}: {path} must be a recognized boolean value, got {value!r}.",
    )


def _integer(value: Any, *, strategy_id: str, path: str) -> int:
    if isinstance(value, bool):
        parsed = None
    elif isinstance(value, int):
        parsed = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        parsed = int(value)
    elif isinstance(value, str):
        raw = value.strip()
        try:
            parsed = int(raw, 10)
        except ValueError:
            parsed = None
    else:
        parsed = None
    if parsed is None:
        raise _runtime_error(
            strategy_id=strategy_id,
            path=path,
            message=f"{strategy_id}: {path} must be an integer, got {value!r}.",
        )
    return parsed


def _utc_timestamp(value: Any, *, strategy_id: str, path: str) -> str | None:
    if value in (None, ""):
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _runtime_error(
            strategy_id=strategy_id,
            path=path,
            message=f"{strategy_id}: {path} must be a valid timestamp, got {value!r}.",
        ) from exc
    if pd.isna(timestamp):
        raise _runtime_error(
            strategy_id=strategy_id,
            path=path,
            message=f"{strategy_id}: {path} must be a valid timestamp, got {value!r}.",
        )
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")


def normalize_v2_runtime_values(
    values: Mapping[str, Any] | None,
    *,
    strategy_id: str = "<unknown strategy>",
    user_boundary: bool = True,
    missing_date_filter: bool = False,
) -> dict[str, Any]:
    """Normalize all four runtime values without truthiness fallbacks.

    ``missing_date_filter=False`` is the legacy/persistence default. The Stage 2
    UI boundary will pass the explicit new-form value instead.
    """

    if values is None:
        source: Mapping[str, Any] = {}
    elif isinstance(values, Mapping):
        source = values
    else:
        raise _runtime_error(
            strategy_id=strategy_id,
            path="runtime",
            message=f"{strategy_id}: runtime values must be a mapping.",
        )
    date_filter = (
        _strict_bool(source["dateFilter"], strategy_id=strategy_id, path="dateFilter")
        if "dateFilter" in source
        else bool(missing_date_filter)
    )
    start = _utc_timestamp(source.get("start"), strategy_id=strategy_id, path="start")
    end = _utc_timestamp(source.get("end"), strategy_id=strategy_id, path="end")
    warmup_field = _RUNTIME_FIELD_BY_NAME["warmupBars"]
    warmup = _integer(
        source["warmupBars"]
        if "warmupBars" in source
        else warmup_field.legacy_default,
        strategy_id=strategy_id,
        path="warmupBars",
    )
    if warmup < 0 or (user_boundary and not 100 <= warmup <= 5000):
        expected = "between 100 and 5000" if user_boundary else "non-negative"
        raise _runtime_error(
            strategy_id=strategy_id,
            path="warmupBars",
            message=f"{strategy_id}: warmupBars must be {expected}, got {warmup}.",
        )
    if date_filter and start is not None and end is not None:
        if pd.Timestamp(end) <= pd.Timestamp(start):
            raise _runtime_error(
                strategy_id=strategy_id,
                path="end",
                message=f"{strategy_id}: end must be later than start when dateFilter is enabled.",
            )
    return {
        "dateFilter": date_filter,
        "start": start,
        "end": end,
        "warmupBars": warmup,
    }


def validate_v2_runtime_declarations(config: Mapping[str, Any]) -> None:
    """Validate optional reserved declarations for V2; V1 is a no-op."""

    if str(config.get("engine", "v1")).strip().lower() != "v2":
        return
    strategy_id = str(config.get("id") or "<unknown strategy>")
    params = config.get("parameters", {})
    if not isinstance(params, Mapping):
        raise _runtime_error(
            strategy_id=strategy_id,
            path="parameters",
            code="V2_INCOMPATIBLE_RUNTIME_DECLARATION",
            message=f"{strategy_id}: parameters must be a mapping.",
        )
    selector = config.get("execution", {})
    selector = selector.get("variantSelector") if isinstance(selector, Mapping) else None
    selector_param = selector.get("param") if isinstance(selector, Mapping) else None
    for name in V2_RESERVED_RUNTIME_PARAM_NAMES:
        option_name = f"{name}_options"
        if option_name in params:
            raise _runtime_error(
                strategy_id=strategy_id,
                path=f"parameters.{option_name}",
                code="V2_INCOMPATIBLE_RUNTIME_DECLARATION",
                message=(
                    f"{strategy_id}: parameters.{option_name} is forbidden; reserved runtime "
                    "fields cannot be supplied through option parameters."
                ),
            )
    for name in V2_RESERVED_RUNTIME_PARAM_NAMES:
        if name not in params:
            continue
        spec = params[name]
        path = f"parameters.{name}"
        expected = _RUNTIME_FIELD_BY_NAME[name]
        if not isinstance(spec, Mapping):
            raise _runtime_error(
                strategy_id=strategy_id,
                path=path,
                code="V2_INCOMPATIBLE_RUNTIME_DECLARATION",
                message=(
                    f"{strategy_id}: {path} must be a mapping compatible with the "
                    f"core runtime contract, got {type(spec).__name__}."
                ),
            )
        reasons: list[str] = []
        role = str(spec.get("role") or "").strip().lower()
        if role != "runtime":
            reasons.append(f"role must be 'runtime', got {spec.get('role')!r}")
        declared_type = str(spec.get("type") or "").strip().lower()
        if declared_type != expected.type:
            reasons.append(f"type must be {expected.type!r}, got {spec.get('type')!r}")

        default = spec.get("default")
        if name == "dateFilter":
            if default is not expected.ui_default:
                reasons.append(f"default must be true, got {default!r}")
        elif name in {"start", "end"}:
            if default not in (None, ""):
                reasons.append(f"default must be empty, got {default!r}")
        else:
            if default != expected.ui_default:
                reasons.append(
                    f"default must be {expected.ui_default}, got {default!r}"
                )
            if spec.get("min") != expected.minimum:
                reasons.append(f"min must be {expected.minimum}, got {spec.get('min')!r}")
            if spec.get("max") != expected.maximum:
                reasons.append(f"max must be {expected.maximum}, got {spec.get('max')!r}")

        raw_optimize = spec.get("optimize", {})
        if "optimize" in spec and not isinstance(raw_optimize, Mapping):
            reasons.append(
                f"optimize must be a mapping when present, got {type(raw_optimize).__name__}"
            )
            optimize: Mapping[str, Any] = {}
        else:
            optimize = raw_optimize
        if optimize.get("enabled", False) is not False:
            reasons.append(
                f"optimize.enabled must be false, got {optimize.get('enabled')!r}"
            )
        if optimize.get("default_enabled", False) is not False:
            reasons.append(
                "optimize.default_enabled must be false, "
                f"got {optimize.get('default_enabled')!r}"
            )
        spec_axis_keys = [key for key in ("gridValues", "values") if key in spec]
        optimize_axis_keys = [
            key
            for key in ("min", "max", "step", "gridValues", "values", "options")
            if key in optimize
        ]
        if spec_axis_keys or optimize_axis_keys:
            reasons.append(
                "optimization axis metadata is forbidden "
                f"(found {spec_axis_keys + optimize_axis_keys!r})"
            )
        if "options" in spec:
            reasons.append("options are forbidden for reserved runtime fields")
        if "depends_on" in spec:
            reasons.append("depends_on is forbidden for reserved runtime fields")
        if selector_param == name:
            reasons.append("reserved runtime fields cannot be variant selectors")

        if reasons:
            raise _runtime_error(
                strategy_id=strategy_id,
                path=path,
                code="V2_INCOMPATIBLE_RUNTIME_DECLARATION",
                message=(
                    f"{strategy_id}: {path} is incompatible with "
                    f"{V2_RUNTIME_CONTRACT_VERSION}: {'; '.join(reasons)}."
                ),
            )

    for child_name, raw_spec in params.items():
        if not isinstance(raw_spec, Mapping) or "depends_on" not in raw_spec:
            continue
        raw_parents = raw_spec.get("depends_on")
        parents = (raw_parents,) if isinstance(raw_parents, str) else tuple(raw_parents or ())
        if str(child_name) in V2_RESERVED_RUNTIME_PARAM_NAMES or any(
            str(parent) in V2_RESERVED_RUNTIME_PARAM_NAMES for parent in parents
        ):
            raise _runtime_error(
                strategy_id=strategy_id,
                path=f"parameters.{child_name}.depends_on",
                code="V2_INCOMPATIBLE_RUNTIME_DECLARATION",
                message=f"{strategy_id}: reserved runtime fields cannot participate in depends_on.",
            )


__all__ = [
    "V2_REBASABLE_DATE_PARAM_NAMES",
    "V2_RESERVED_RUNTIME_PARAM_NAMES",
    "V2_RUNTIME_CONTRACT_VERSION",
    "V2_RUNTIME_FIELDS",
    "V2RuntimeField",
    "V2RuntimeValidationError",
    "normalize_v2_runtime_values",
    "runtime_contract_payload",
    "validate_v2_runtime_declarations",
]
