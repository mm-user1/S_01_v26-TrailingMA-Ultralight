"""Strict raw-source and authoritative window preparation for Strategy Lab."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from core.backtest_engine import load_data, prepare_dataset_with_warmup
from core.engine_v2.runtime_contract import normalize_v2_runtime_field_value
from core.walkforward_engine import WindowSplit, build_calendar_month_windows

from .config import RunSpec
from .inventory import EXPECTED_HEADER, InventoryError, parse_filename


QUALITY_COLUMNS = (
    "canonical_symbol",
    "relative_filename",
    "cell",
    "window_id",
    "segment",
    "status",
    "first_timestamp",
    "last_timestamp",
    "row_count",
    "expected_row_count",
    "warmup_row_count",
    "trade_start_idx",
    "source_sha256",
    "message",
)
_UNIX_SECONDS_RE = re.compile(r"[+-]?[0-9]+")


class DataQualityError(ValueError):
    """A deterministic hard failure in source or segment validation."""


@dataclass(frozen=True)
class SourceSnapshot:
    size: int
    mtime_ns: int
    sha256: str


@dataclass(frozen=True)
class ValidatedSource:
    entry: Mapping[str, Any]
    path: Path
    dataframe: pd.DataFrame
    snapshot: SourceSnapshot
    first_timestamp: str
    last_timestamp: str
    row_count: int


@dataclass(frozen=True)
class PreparedSegment:
    window_id: int
    segment: str
    dataframe: pd.DataFrame
    trade_start_idx: int
    evaluation_row_count: int
    first_timestamp: str
    last_timestamp: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_second(raw: str, field: str) -> tuple[int, str]:
    value = raw.strip()
    if _UNIX_SECONDS_RE.fullmatch(value) is None:
        raise DataQualityError(f"{field}: expected an integer Unix-second timestamp.")
    seconds = int(value)
    if abs(seconds) >= 100_000_000_000:
        raise DataQualityError(
            f"{field}: numeric timestamps must use Unix seconds; milliseconds are unsupported."
        )
    try:
        parsed = datetime.fromtimestamp(seconds, tz=UTC)
    except (OverflowError, OSError, ValueError):
        raise DataQualityError(f"{field}: invalid Unix-second timestamp {value!r}.") from None
    return seconds, parsed.isoformat().replace("+00:00", "Z")


def _finite_number(raw: str, field: str, *, positive: bool, non_negative: bool = False) -> float:
    try:
        value = float(raw)
    except ValueError:
        raise DataQualityError(f"{field}: expected a finite number.") from None
    if not math.isfinite(value):
        raise DataQualityError(f"{field}: expected a finite number.")
    if positive and value <= 0.0:
        raise DataQualityError(f"{field}: expected a positive value.")
    if non_negative and value < 0.0:
        raise DataQualityError(f"{field}: expected a non-negative value.")
    return value


def quality_row(
    entry: Mapping[str, Any],
    *,
    window_id: int | str = "",
    segment: str = "source",
    status: str,
    first_timestamp: str = "",
    last_timestamp: str = "",
    row_count: int | str = "",
    expected_row_count: int | str = "",
    warmup_row_count: int | str = "",
    trade_start_idx: int | str = "",
    message: str = "",
) -> dict[str, Any]:
    return {
        "canonical_symbol": entry["canonical_symbol"],
        "relative_filename": entry["relative_filename"],
        "cell": entry["cell"],
        "window_id": window_id,
        "segment": segment,
        "status": status,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "row_count": row_count,
        "expected_row_count": expected_row_count,
        "warmup_row_count": warmup_row_count,
        "trade_start_idx": trade_start_idx,
        "source_sha256": entry["sha256"],
        "message": message,
    }


def validate_source(
    data_root: Path,
    entry: Mapping[str, Any],
    *,
    timeframe_minutes: int,
) -> ValidatedSource:
    """Validate raw rows without sorting or coercion, then load execution data."""

    root = data_root.resolve()
    relative = str(entry["relative_filename"])
    path = root / relative
    if Path(relative).name != relative or Path(relative).is_absolute():
        raise DataQualityError(f"{relative}: inventory relative filename is invalid.")
    if path.is_symlink():
        raise DataQualityError(f"{relative}: source symlinks are not allowed.")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError):
        raise DataQualityError(f"{relative}: source must be a regular file within the data root.") from None
    if not resolved.is_file():
        raise DataQualityError(f"{relative}: source must be a regular file.")

    try:
        parsed = parse_filename(relative, expected_timeframe_minutes=timeframe_minutes)
    except InventoryError as exc:
        raise DataQualityError(f"{relative}: {exc}") from None
    identity = {
        "exchange": parsed.exchange,
        "canonical_symbol": parsed.canonical_symbol,
        "raw_instrument_label": parsed.raw_instrument_label,
        "filename_start": parsed.start,
        "filename_end": parsed.end,
    }
    for field, observed in identity.items():
        if entry[field] != observed:
            raise DataQualityError(f"{relative}: {field} does not match the accepted inventory entry.")

    stat = resolved.stat()
    digest = _sha256_file(resolved)
    snapshot = SourceSnapshot(size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest)
    if snapshot.size != entry["file_size"]:
        raise DataQualityError(f"{relative}: file size does not match the accepted inventory entry.")
    if snapshot.sha256 != entry["sha256"]:
        raise DataQualityError(f"{relative}: SHA-256 does not match the accepted inventory entry.")

    timestamps: list[int] = []
    normalized_timestamps: list[str] = []
    try:
        with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header != EXPECTED_HEADER:
                raise DataQualityError(
                    f"{relative}: header must be {','.join(EXPECTED_HEADER)}."
                )
            for line_number, row in enumerate(reader, start=2):
                if len(row) != len(EXPECTED_HEADER):
                    raise DataQualityError(
                        f"{relative}:{line_number}: expected {len(EXPECTED_HEADER)} columns."
                    )
                seconds, normalized = _utc_second(
                    row[0], f"{relative}:{line_number}.time"
                )
                open_value = _finite_number(
                    row[1], f"{relative}:{line_number}.open", positive=True
                )
                high = _finite_number(
                    row[2], f"{relative}:{line_number}.high", positive=True
                )
                low = _finite_number(
                    row[3], f"{relative}:{line_number}.low", positive=True
                )
                close = _finite_number(
                    row[4], f"{relative}:{line_number}.close", positive=True
                )
                _finite_number(
                    row[5], f"{relative}:{line_number}.Volume", positive=False, non_negative=True
                )
                if high < max(open_value, close, low):
                    raise DataQualityError(
                        f"{relative}:{line_number}: high must be >= max(open, close, low)."
                    )
                if low > min(open_value, close, high):
                    raise DataQualityError(
                        f"{relative}:{line_number}: low must be <= min(open, close, high)."
                    )
                timestamps.append(seconds)
                normalized_timestamps.append(normalized)
    except UnicodeError as exc:
        raise DataQualityError(f"{relative}: invalid UTF-8: {exc}") from None

    row_count = len(timestamps)
    if row_count != entry["row_count"]:
        raise DataQualityError(
            f"{relative}: raw row count {row_count} does not match inventory {entry['row_count']}."
        )
    if not timestamps:
        raise DataQualityError(f"{relative}: source has no data rows.")
    if len(set(timestamps)) != row_count:
        raise DataQualityError(f"{relative}: duplicate raw timestamp detected.")
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise DataQualityError(f"{relative}: raw timestamps must be strictly increasing.")
    cadence = timeframe_minutes * 60
    if any(right - left != cadence for left, right in zip(timestamps, timestamps[1:])):
        raise DataQualityError(
            f"{relative}: raw timestamp cadence must be exactly {cadence} seconds."
        )
    first_timestamp = normalized_timestamps[0]
    last_timestamp = normalized_timestamps[-1]
    if first_timestamp != entry["first_timestamp"]:
        raise DataQualityError(f"{relative}: first timestamp does not match inventory.")
    if last_timestamp != entry["last_timestamp"]:
        raise DataQualityError(f"{relative}: last timestamp does not match inventory.")

    dataframe = load_data(resolved)
    if len(dataframe) != row_count:
        raise DataQualityError(f"{relative}: loaded dataframe row count changed after the raw gate.")
    return ValidatedSource(
        entry=entry,
        path=resolved,
        dataframe=dataframe,
        snapshot=snapshot,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        row_count=row_count,
    )


def validate_selected_sources(
    data_root: Path,
    entries: Sequence[Mapping[str, Any]],
    *,
    timeframe_minutes: int,
) -> tuple[tuple[ValidatedSource, ...], tuple[dict[str, Any], ...]]:
    validated: list[ValidatedSource] = []
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for entry in entries:
        try:
            source = validate_source(
                data_root,
                entry,
                timeframe_minutes=timeframe_minutes,
            )
        except DataQualityError as exc:
            message = str(exc)
            failures.append(message)
            rows.append(
                quality_row(
                    entry,
                    status="error",
                    expected_row_count=entry["row_count"],
                    message=message,
                )
            )
        else:
            validated.append(source)
            rows.append(
                quality_row(
                    entry,
                    status="ok",
                    first_timestamp=source.first_timestamp,
                    last_timestamp=source.last_timestamp,
                    row_count=source.row_count,
                    expected_row_count=entry["row_count"],
                    message="raw source validated",
                )
            )
    if failures:
        error = DataQualityError("source validation failed: " + " | ".join(failures))
        error.quality_rows = tuple(rows)  # type: ignore[attr-defined]
        raise error
    return tuple(validated), tuple(rows)


def _window_tuple(window: WindowSplit) -> tuple[int, int, int, int, int]:
    return (
        int(window.window_id),
        int(pd.Timestamp(window.is_start).value),
        int(pd.Timestamp(window.is_end).value),
        int(pd.Timestamp(window.oos_start).value),
        int(pd.Timestamp(window.oos_end).value),
    )


def expected_complete_oos_end(
    window: WindowSplit,
    *,
    oos_period_months: int,
    timeframe_minutes: int,
) -> pd.Timestamp:
    """Return and validate the inclusive final bar of a complete OOS interval."""

    exclusive_end = pd.Timestamp(window.oos_start) + pd.DateOffset(
        months=oos_period_months
    )
    expected_end = exclusive_end - pd.Timedelta(minutes=timeframe_minutes)
    if pd.Timestamp(window.oos_end) != expected_end:
        raise DataQualityError(
            f"window {window.window_id} OOS is truncated; expected final bar "
            f"{expected_end.isoformat()}, got {pd.Timestamp(window.oos_end).isoformat()}."
        )
    return expected_end


def build_authoritative_windows(
    spec: RunSpec,
    sources: Sequence[ValidatedSource],
) -> tuple[WindowSplit, ...]:
    windows_contract = spec.generation["windows"]
    requested_start = windows_contract["requested_start"]
    normalized_end = normalize_v2_runtime_field_value(
        "end", windows_contract["requested_end"]
    )
    if normalized_end is None:
        raise DataQualityError("generation.windows.requested_end is required.")
    reference: tuple[WindowSplit, ...] | None = None
    reference_tuples: tuple[tuple[int, int, int, int, int], ...] | None = None
    for source in sources:
        first = source.dataframe.index[0]
        last = source.dataframe.index[-1]
        requested_start_ts = pd.Timestamp(requested_start, tz="UTC")
        requested_end_ts = pd.Timestamp(normalized_end)
        if requested_start_ts < first or requested_end_ts.normalize() > last.normalize():
            raise DataQualityError(
                f"{source.entry['canonical_symbol']}: requested range is outside verified source coverage."
            )
        with redirect_stdout(io.StringIO()):
            built = tuple(
                build_calendar_month_windows(
                    source.dataframe.index,
                    requested_start,
                    requested_end_ts,
                    windows_contract["is_period_months"],
                    windows_contract["oos_period_months"],
                )
            )
        if len(built) != windows_contract["expected_window_count"]:
            raise DataQualityError(
                "authoritative calendar builder count does not match expected_window_count."
            )
        if [window.window_id for window in built] != list(range(1, len(built) + 1)):
            raise DataQualityError("authoritative calendar window IDs are not sequential.")
        for window in built:
            expected_complete_oos_end(
                window,
                oos_period_months=int(windows_contract["oos_period_months"]),
                timeframe_minutes=int(windows_contract["timeframe_minutes"]),
            )
        tuples = tuple(_window_tuple(window) for window in built)
        if reference is None:
            reference = built
            reference_tuples = tuples
        elif tuples != reference_tuples:
            raise DataQualityError(
                f"{source.entry['canonical_symbol']}: authoritative window boundaries differ across selected tickers."
            )
    if reference is None:
        raise DataQualityError("at least one validated source is required.")
    return reference


def prepare_segment(
    source: ValidatedSource,
    window: WindowSplit,
    segment: str,
    *,
    warmup_bars: int,
    timeframe_minutes: int,
    oos_period_months: int | None = None,
) -> PreparedSegment:
    if segment == "is":
        start, end = window.is_start, window.is_end
    elif segment == "oos":
        start, end = window.oos_start, window.oos_end
        if oos_period_months is None:
            raise DataQualityError("OOS preparation requires the declared OOS month count.")
        expected_end = expected_complete_oos_end(
            window,
            oos_period_months=oos_period_months,
            timeframe_minutes=timeframe_minutes,
        )
    else:
        raise DataQualityError(f"unsupported segment {segment!r}.")
    prepared, trade_start_idx = prepare_dataset_with_warmup(
        source.dataframe,
        pd.Timestamp(start),
        pd.Timestamp(end),
        warmup_bars,
    )
    expected_eval = int((pd.Timestamp(end) - pd.Timestamp(start)).total_seconds()) // (
        timeframe_minutes * 60
    ) + 1
    if trade_start_idx != warmup_bars:
        raise DataQualityError(
            f"{source.entry['canonical_symbol']} window {window.window_id} {segment}: "
            f"insufficient warmup; expected {warmup_bars}, got {trade_start_idx}."
        )
    if len(prepared) != warmup_bars + expected_eval:
        raise DataQualityError(
            f"{source.entry['canonical_symbol']} window {window.window_id} {segment}: "
            "prepared length does not equal warmup plus evaluation rows."
        )
    if prepared.empty or prepared.index[trade_start_idx] != pd.Timestamp(start):
        raise DataQualityError(
            f"{source.entry['canonical_symbol']} window {window.window_id} {segment}: "
            "evaluation first timestamp does not match WindowSplit."
        )
    if prepared.index[-1] != pd.Timestamp(end):
        raise DataQualityError(
            f"{source.entry['canonical_symbol']} window {window.window_id} {segment}: "
            "evaluation final timestamp does not match WindowSplit."
        )
    if segment == "oos" and prepared.index[-1] != expected_end:
        raise DataQualityError(
            f"{source.entry['canonical_symbol']} window {window.window_id} OOS: "
            "prepared final timestamp is not the complete declared OOS final bar."
        )
    return PreparedSegment(
        window_id=window.window_id,
        segment=segment,
        dataframe=prepared,
        trade_start_idx=trade_start_idx,
        evaluation_row_count=expected_eval,
        first_timestamp=pd.Timestamp(start).isoformat().replace("+00:00", "Z"),
        last_timestamp=pd.Timestamp(end).isoformat().replace("+00:00", "Z"),
    )


def verify_source_preservation(sources: Sequence[ValidatedSource]) -> dict[str, Any]:
    for source in sources:
        current_stat = source.path.stat()
        current_hash = _sha256_file(source.path)
        if (
            current_stat.st_size != source.snapshot.size
            or current_stat.st_mtime_ns != source.snapshot.mtime_ns
            or current_hash != source.snapshot.sha256
        ):
            raise DataQualityError(
                f"{source.entry['relative_filename']}: source hash, size, or mtime changed during generation."
            )
    return {
        "verified": True,
        "source_count": len(sources),
        "sha256_size_mtime_unchanged": True,
    }


__all__ = [
    "DataQualityError",
    "PreparedSegment",
    "QUALITY_COLUMNS",
    "ValidatedSource",
    "build_authoritative_windows",
    "expected_complete_oos_end",
    "prepare_segment",
    "quality_row",
    "validate_selected_sources",
    "validate_source",
    "verify_source_preservation",
]
