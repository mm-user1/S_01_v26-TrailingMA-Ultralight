"""Versioned persistence metadata for core-owned Backtester V2 runtime values."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

from .diagnostics import (
    V2Diagnostic,
    V2ValidationError,
    diagnostic_from_mapping,
    warning_messages,
)
from .runtime_contract import (
    V2_RUNTIME_CONTRACT_VERSION,
    normalize_v2_runtime_field_value,
    normalize_v2_runtime_values,
)


V2_RUNTIME_METADATA_SCHEMA_VERSION = "v2_runtime_metadata_v1"
_VERSIONED_METADATA_SCHEMA_RE = re.compile(r"^v2_runtime_metadata_v\d+$")
_METADATA_KEYS = frozenset(
    {
        "schema_version",
        "contract_version",
        "values",
        "diagnostics",
        "validation_warnings",
    }
)
_DIAGNOSTIC_KEYS = frozenset(
    {"severity", "code", "strategy_id", "path", "variant", "message"}
)


@dataclass(frozen=True, slots=True)
class V2RuntimeMetadataResolution:
    """Typed result of resolving current or legacy stored V2 runtime facts."""

    values: dict[str, Any] | None
    source: str
    diagnostics: tuple[V2Diagnostic, ...]
    validation_warnings: tuple[str, ...]
    usable: bool


def _metadata_diagnostic(
    *, strategy_id: str, path: str, message: str
) -> V2Diagnostic:
    return V2Diagnostic(
        severity="warning",
        code="V2_STORED_RUNTIME_METADATA_INCOMPATIBLE",
        strategy_id=strategy_id,
        path=path,
        variant=None,
        message=message,
    )


def _metadata_build_error(
    *, strategy_id: str, path: str, message: str
) -> V2ValidationError:
    return V2ValidationError(
        V2Diagnostic(
            severity="error",
            code="V2_RUNTIME_METADATA_INVALID",
            strategy_id=strategy_id,
            path=path,
            variant=None,
            message=f"{strategy_id}: {message}",
        )
    )


def parse_stored_config_json(
    raw: Any,
    *,
    strategy_id: str = "<unknown strategy>",
    present: bool = True,
) -> tuple[dict[str, Any] | None, V2Diagnostic | None]:
    """Parse one stored config value without raising at the viewing boundary."""

    if not present:
        return {}, None
    if raw is None or isinstance(raw, str) and not raw.strip():
        return {}, None
    if isinstance(raw, Mapping):
        return dict(raw), None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            detail = f"invalid JSON: {exc.msg}"
        else:
            if isinstance(parsed, Mapping):
                return dict(parsed), None
            detail = "JSON value must be an object"
    else:
        detail = "value must be a mapping or a JSON object string"
    diagnostic = _metadata_diagnostic(
        strategy_id=strategy_id,
        path="config_json",
        message=f"{strategy_id}: stored config_json is incompatible: {detail}.",
    )
    return None, diagnostic


def _normalize_diagnostics(
    diagnostics: Iterable[V2Diagnostic | Mapping[str, Any]],
    *,
    strategy_id: str = "<unknown strategy>",
) -> tuple[V2Diagnostic, ...]:
    normalized: list[V2Diagnostic] = []
    for item in diagnostics:
        if isinstance(item, V2Diagnostic):
            diagnostic = item
        elif isinstance(item, Mapping):
            if set(item) != _DIAGNOSTIC_KEYS:
                raise _metadata_build_error(
                    strategy_id=strategy_id,
                    path="v2_runtime.diagnostics",
                    message="V2 runtime diagnostics must use the exact structured fields.",
                )
            if not all(
                isinstance(item.get(name), str)
                for name in ("severity", "code", "strategy_id", "path", "message")
            ) or item.get("variant") is not None and not isinstance(
                item.get("variant"), str
            ):
                raise _metadata_build_error(
                    strategy_id=strategy_id,
                    path="v2_runtime.diagnostics",
                    message="V2 runtime diagnostic field types are invalid.",
                )
            diagnostic = diagnostic_from_mapping(item)
        else:
            raise _metadata_build_error(
                strategy_id=strategy_id,
                path="v2_runtime.diagnostics",
                message="V2 runtime diagnostics must contain mappings.",
            )
        if diagnostic.severity not in {"info", "warning", "error"}:
            raise _metadata_build_error(
                strategy_id=strategy_id,
                path="v2_runtime.diagnostics",
                message=f"Unsupported V2 runtime diagnostic severity: {diagnostic.severity!r}.",
            )
        normalized.append(diagnostic)
    return tuple(normalized)


def build_v2_runtime_metadata(
    values: Mapping[str, Any],
    diagnostics: Iterable[V2Diagnostic | Mapping[str, Any]] = (),
    *,
    strategy_id: str = "<unknown strategy>",
) -> dict[str, Any]:
    """Build fresh JSON-safe metadata from an already canonical complete mapping."""

    ordered_names = ("dateFilter", "start", "end", "warmupBars")
    if not isinstance(values, Mapping) or set(values) != set(ordered_names):
        raise _metadata_build_error(
            strategy_id=strategy_id,
            path="v2_runtime.values",
            message="V2 runtime metadata values must be complete and canonical.",
        )
    canonical: dict[str, Any] = {}
    for name in ordered_names:
        raw_value = values[name]
        try:
            normalized = normalize_v2_runtime_field_value(
                name,
                raw_value,
                strategy_id=strategy_id,
                path=f"v2_runtime.values.{name}",
                user_boundary=False,
            )
        except V2ValidationError as exc:
            raise _metadata_build_error(
                strategy_id=strategy_id,
                path=f"v2_runtime.values.{name}",
                message=f"V2 runtime metadata value is invalid: {exc}.",
            ) from exc
        if type(raw_value) is not type(normalized) or raw_value != normalized:
            raise _metadata_build_error(
                strategy_id=strategy_id,
                path=f"v2_runtime.values.{name}",
                message="V2 runtime metadata values must be complete and canonical.",
            )
        canonical[name] = normalized
    if (
        canonical["dateFilter"]
        and canonical["start"] is not None
        and canonical["end"] is not None
        and pd.Timestamp(canonical["end"]) <= pd.Timestamp(canonical["start"])
    ):
        raise _metadata_build_error(
            strategy_id=strategy_id,
            path="v2_runtime.values.end",
            message="V2 runtime metadata end must be later than start.",
        )
    normalized_diagnostics = _normalize_diagnostics(
        diagnostics, strategy_id=strategy_id
    )
    if any(item.severity == "error" for item in normalized_diagnostics):
        raise _metadata_build_error(
            strategy_id=strategy_id,
            path="v2_runtime.diagnostics",
            message="V2 runtime metadata cannot persist fatal diagnostics.",
        )
    payload = {
        "schema_version": V2_RUNTIME_METADATA_SCHEMA_VERSION,
        "contract_version": V2_RUNTIME_CONTRACT_VERSION,
        "values": dict(canonical),
        "diagnostics": [item.to_dict() for item in normalized_diagnostics],
        "validation_warnings": list(warning_messages(normalized_diagnostics)),
    }
    # Validate JSON safety and return data detached from caller-owned objects.
    return json.loads(json.dumps(payload, allow_nan=False))


def parse_v2_runtime_metadata(
    raw: Any,
    *,
    strategy_id: str = "<unknown strategy>",
) -> V2RuntimeMetadataResolution:
    """Parse only the current exact metadata envelope; never consult legacy facts."""

    path = "config_json.v2_runtime"
    try:
        if not isinstance(raw, Mapping) or set(raw) != _METADATA_KEYS:
            raise ValueError("metadata must be a mapping with the exact current fields")
        if raw.get("schema_version") != V2_RUNTIME_METADATA_SCHEMA_VERSION:
            raise ValueError("metadata schema_version is unsupported")
        if raw.get("contract_version") != V2_RUNTIME_CONTRACT_VERSION:
            raise ValueError("metadata contract_version is unsupported")
        raw_diagnostics = raw.get("diagnostics")
        if not isinstance(raw_diagnostics, list):
            raise ValueError("diagnostics must be a list")
        diagnostics = _normalize_diagnostics(
            raw_diagnostics, strategy_id=strategy_id
        )
        if any(item.severity == "error" for item in diagnostics):
            raise ValueError("metadata contains a fatal diagnostic")
        raw_warnings = raw.get("validation_warnings")
        if not isinstance(raw_warnings, list) or not all(
            isinstance(item, str) for item in raw_warnings
        ):
            raise ValueError("validation_warnings must be a list of strings")
        if tuple(raw_warnings) != warning_messages(diagnostics):
            raise ValueError("validation_warnings does not match diagnostics")
        rebuilt = build_v2_runtime_metadata(
            raw.get("values"), diagnostics, strategy_id=strategy_id
        )
        return V2RuntimeMetadataResolution(
            values=rebuilt["values"],
            source="current",
            diagnostics=diagnostics,
            validation_warnings=tuple(rebuilt["validation_warnings"]),
            usable=True,
        )
    except (TypeError, ValueError) as exc:
        diagnostic = _metadata_diagnostic(
            strategy_id=strategy_id,
            path=path,
            message=f"{strategy_id}: stored V2 runtime metadata is incompatible: {exc}.",
        )
        return V2RuntimeMetadataResolution(
            values=None,
            source="unavailable",
            diagnostics=(diagnostic,),
            validation_warnings=(diagnostic.message,),
            usable=False,
        )


def resolve_stored_v2_runtime(
    study: Mapping[str, Any],
    *,
    strategy_id: str | None = None,
) -> V2RuntimeMetadataResolution:
    """Resolve stored V2 runtime facts with current/legacy/default precedence."""

    resolved_strategy_id = str(
        strategy_id or study.get("strategy_id") or "<unknown strategy>"
    )
    raw_config = study.get("config_json")
    config, config_diagnostic = parse_stored_config_json(
        raw_config,
        strategy_id=resolved_strategy_id,
        present="config_json" in study,
    )
    if config is None:
        assert config_diagnostic is not None
        return V2RuntimeMetadataResolution(
            values=None,
            source="unavailable",
            diagnostics=(config_diagnostic,),
            validation_warnings=(config_diagnostic.message,),
            usable=False,
        )
    metadata_present = "v2_runtime" in config
    metadata_resolution: V2RuntimeMetadataResolution | None = None
    if metadata_present:
        metadata_resolution = parse_v2_runtime_metadata(
            config.get("v2_runtime"), strategy_id=resolved_strategy_id
        )
        if metadata_resolution.usable:
            return metadata_resolution
        raw_metadata = config.get("v2_runtime")
        if isinstance(raw_metadata, Mapping):
            schema_version = raw_metadata.get("schema_version")
            contract_version = raw_metadata.get("contract_version")
            unsupported_schema = (
                isinstance(schema_version, str)
                and _VERSIONED_METADATA_SCHEMA_RE.fullmatch(schema_version) is not None
                and schema_version != V2_RUNTIME_METADATA_SCHEMA_VERSION
            )
            unsupported_contract = (
                "contract_version" in raw_metadata
                and contract_version != V2_RUNTIME_CONTRACT_VERSION
            )
            if unsupported_schema or unsupported_contract:
                return metadata_resolution

    fixed_raw = config.get("fixed_params")
    fixed_params = fixed_raw if isinstance(fixed_raw, Mapping) else {}
    legacy: dict[str, Any] = {}
    has_legacy_fact = False
    for name in ("dateFilter", "start", "end"):
        if name in fixed_params:
            legacy[name] = fixed_params[name]
            has_legacy_fact = True
    if "warmup_bars" in study and study.get("warmup_bars") is not None:
        legacy["warmupBars"] = study.get("warmup_bars")
        has_legacy_fact = True
    elif "warmup_bars" in config and config.get("warmup_bars") is not None:
        legacy["warmupBars"] = config.get("warmup_bars")
        has_legacy_fact = True

    prior_diagnostics = (
        metadata_resolution.diagnostics
        if metadata_resolution is not None
        else ()
    )
    if has_legacy_fact:
        try:
            values = normalize_v2_runtime_values(
                legacy,
                strategy_id=resolved_strategy_id,
                user_boundary=False,
                missing_date_filter=False,
            )
        except ValueError as exc:
            diagnostic = _metadata_diagnostic(
                strategy_id=resolved_strategy_id,
                path="config_json.fixed_params",
                message=f"{resolved_strategy_id}: stored legacy V2 runtime facts are incompatible: {exc}.",
            )
            diagnostics = (*prior_diagnostics, diagnostic)
            return V2RuntimeMetadataResolution(
                values=None,
                source="unavailable",
                diagnostics=diagnostics,
                validation_warnings=warning_messages(diagnostics),
                usable=False,
            )
        return V2RuntimeMetadataResolution(
            values=values,
            source="legacy",
            diagnostics=prior_diagnostics,
            validation_warnings=warning_messages(prior_diagnostics),
            usable=True,
        )

    if metadata_present:
        assert metadata_resolution is not None
        return metadata_resolution

    values = normalize_v2_runtime_values(
        None,
        strategy_id=resolved_strategy_id,
        user_boundary=False,
        missing_date_filter=False,
    )
    return V2RuntimeMetadataResolution(
        values=values,
        source="defaulted",
        diagnostics=(),
        validation_warnings=(),
        usable=True,
    )


__all__ = [
    "V2_RUNTIME_METADATA_SCHEMA_VERSION",
    "V2RuntimeMetadataResolution",
    "build_v2_runtime_metadata",
    "parse_v2_runtime_metadata",
    "parse_stored_config_json",
    "resolve_stored_v2_runtime",
]
