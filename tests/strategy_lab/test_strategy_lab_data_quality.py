from __future__ import annotations

import copy
import os
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from tools.strategy_lab.config import load_run_spec
from tools.strategy_lab.data_quality import (
    DataQualityError,
    SourceSnapshot,
    ValidatedSource,
    build_authoritative_windows,
    prepare_segment,
    validate_selected_sources,
    validate_source,
    verify_source_preservation,
)

from tests.strategy_lab.phase1a_helpers import (
    CURRENT_RUN_SPEC,
    build_small_inventory,
    refresh_entry_file_facts,
    write_source,
)


def _small_pack(tmp_path: Path) -> tuple[Path, dict, Path]:
    data_root = tmp_path / "market"
    data_root.mkdir()
    target = write_source(data_root, "OKX", "AAAUSDT")
    write_source(data_root, "BYBIT", "ZZZUSDT")
    inventory = build_small_inventory(data_root)
    entry = next(
        copy.deepcopy(value)
        for value in inventory["entries"]
        if value["canonical_symbol"] == "AAAUSDT"
    )
    return data_root, entry, target


def _rewrite(path: Path, mutator) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    mutator(lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _field(lines: list[str], row: int, column: int, value: str) -> None:
    values = lines[row].split(",")
    values[column] = value
    lines[row] = ",".join(values)


def test_valid_raw_source_is_loaded_only_after_the_strict_gate(tmp_path, monkeypatch):
    data_root, entry, _ = _small_pack(tmp_path)
    calls = []
    from tools.strategy_lab import data_quality

    original = data_quality.load_data

    def counted(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(data_quality, "load_data", counted)
    source = validate_source(data_root, entry, timeframe_minutes=30)

    assert source.row_count == 48
    assert len(calls) == 1
    assert source.first_timestamp == "2025-08-01T00:00:00Z"
    assert source.last_timestamp == "2025-08-01T23:30:00Z"


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("fractional_timestamp", "integer Unix-second"),
        ("millisecond_timestamp", "milliseconds are unsupported"),
        ("non_monotonic", "strictly increasing"),
        ("duplicate", "duplicate raw timestamp"),
        ("gap", "cadence must be exactly"),
        ("masked_gap", "duplicate raw timestamp"),
        ("wrong_first", "first timestamp"),
        ("wrong_last", "last timestamp"),
        ("row_count", "raw row count"),
        ("header", "header must be"),
        ("nan_ohlc", "finite number"),
        ("infinite_ohlc", "finite number"),
        ("nonpositive_ohlc", "positive value"),
        ("bad_high", "high must be"),
        ("bad_low", "low must be"),
        ("negative_volume", "non-negative value"),
        ("infinite_volume", "finite number"),
    ],
)
def test_raw_row_failures_are_detected_before_load_data(
    tmp_path, monkeypatch, case, message
):
    data_root, entry, target = _small_pack(tmp_path)

    def mutate(lines: list[str]) -> None:
        if case == "fractional_timestamp":
            _field(lines, 1, 0, "1754006400.5")
        elif case == "millisecond_timestamp":
            _field(lines, 1, 0, "1754006400000")
        elif case == "non_monotonic":
            first, second = lines[1].split(","), lines[2].split(",")
            first[0], second[0] = second[0], first[0]
            lines[1], lines[2] = ",".join(first), ",".join(second)
        elif case in {"duplicate", "masked_gap"}:
            _field(lines, 3, 0, lines[2].split(",")[0])
        elif case == "gap":
            _field(lines, 3, 0, str(int(lines[3].split(",")[0]) + 60))
        elif case == "wrong_first":
            for row in range(1, len(lines)):
                _field(lines, row, 0, str(int(lines[row].split(",")[0]) - 1800))
        elif case == "wrong_last":
            for row in range(1, len(lines)):
                _field(lines, row, 0, str(int(lines[row].split(",")[0]) + 1800))
        elif case == "row_count":
            lines.pop()
        elif case == "header":
            lines[0] = "time,open,high,low,close,volume"
        elif case == "nan_ohlc":
            _field(lines, 1, 1, "NaN")
        elif case == "infinite_ohlc":
            _field(lines, 1, 2, "Infinity")
        elif case == "nonpositive_ohlc":
            _field(lines, 1, 4, "0")
        elif case == "bad_high":
            _field(lines, 1, 2, "1")
        elif case == "bad_low":
            _field(lines, 1, 3, "100.1")
        elif case == "negative_volume":
            _field(lines, 1, 5, "-1")
        elif case == "infinite_volume":
            _field(lines, 1, 5, "-Infinity")

    _rewrite(target, mutate)
    refresh_entry_file_facts(entry, target)
    if case == "wrong_last":
        entry["first_timestamp"] = "2025-08-01T00:30:00Z"
    called = False

    def forbidden(_path):
        nonlocal called
        called = True
        raise AssertionError("load_data must not run")

    monkeypatch.setattr("tools.strategy_lab.data_quality.load_data", forbidden)
    with pytest.raises(DataQualityError, match=message):
        validate_source(data_root, entry, timeframe_minutes=30)
    assert called is False


@pytest.mark.parametrize(("field", "message"), [("file_size", "file size"), ("sha256", "SHA-256")])
def test_source_size_and_sha_are_bound_to_inventory(tmp_path, field, message):
    data_root, entry, _ = _small_pack(tmp_path)
    entry[field] = 0 if field == "file_size" else "0" * 64
    with pytest.raises(DataQualityError, match=message):
        validate_source(data_root, entry, timeframe_minutes=30)


def test_all_selected_sources_are_checked_and_quality_rows_are_deterministic(tmp_path):
    data_root, first, target = _small_pack(tmp_path)
    inventory = build_small_inventory(data_root)
    entries = [copy.deepcopy(entry) for entry in inventory["entries"]]
    target.write_bytes(target.read_bytes() + b"\n")

    messages = []
    rows = []
    for _ in range(2):
        with pytest.raises(DataQualityError) as caught:
            validate_selected_sources(data_root, entries, timeframe_minutes=30)
        messages.append(str(caught.value))
        rows.append(caught.value.quality_rows)

    assert messages[0] == messages[1]
    assert rows[0] == rows[1]
    assert len(rows[0]) == 2
    assert {row["status"] for row in rows[0]} == {"ok", "error"}


def _source_for_index(index: pd.DatetimeIndex, symbol: str = "AAAUSDT") -> ValidatedSource:
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "Volume": 1.0,
        },
        index=index,
    )
    return ValidatedSource(
        entry={
            "canonical_symbol": symbol,
            "relative_filename": f"{symbol}.csv",
            "cell": "dev",
            "sha256": "0" * 64,
        },
        path=Path(f"{symbol}.csv"),
        dataframe=frame,
        snapshot=SourceSnapshot(0, 0, "0" * 64),
        first_timestamp=index[0].isoformat().replace("+00:00", "Z"),
        last_timestamp=index[-1].isoformat().replace("+00:00", "Z"),
        row_count=len(index),
    )


def test_authoritative_windows_reuse_builder_and_preserve_frozen_pins(monkeypatch):
    spec = load_run_spec(CURRENT_RUN_SPEC, validate_inventory=False)
    index = pd.date_range("2025-08-01", "2026-08-01 23:30", freq="30min", tz="UTC")
    source = _source_for_index(index)
    from tools.strategy_lab import data_quality

    original = data_quality.build_calendar_month_windows
    calls = []

    def counted(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(data_quality, "build_calendar_month_windows", counted)
    windows = build_authoritative_windows(spec, (source,))

    assert len(calls) == 1
    assert len(windows) == 8
    assert windows[0].is_start == pd.Timestamp("2025-10-01T00:00:00Z")
    assert windows[0].oos_end == pd.Timestamp("2025-12-31T23:30:00Z")
    assert windows[-1].is_start == pd.Timestamp("2026-05-01T00:00:00Z")
    assert windows[-1].oos_end == pd.Timestamp("2026-07-31T23:30:00Z")


def test_window_count_coverage_and_cross_ticker_boundaries_are_strict(monkeypatch):
    spec = load_run_spec(CURRENT_RUN_SPEC, validate_inventory=False)
    full = pd.date_range("2025-08-01", "2026-08-01 23:30", freq="30min", tz="UTC")
    short = pd.date_range("2025-10-02", "2026-08-01 23:30", freq="30min", tz="UTC")
    with pytest.raises(DataQualityError, match="outside verified source coverage"):
        build_authoritative_windows(spec, (_source_for_index(short),))

    changed_generation = copy.deepcopy(spec.generation)
    changed_generation["windows"]["expected_window_count"] = 7
    changed = replace(spec, generation=changed_generation)
    with pytest.raises(DataQualityError, match="count"):
        build_authoritative_windows(changed, (_source_for_index(full),))

    from tools.strategy_lab import data_quality

    original = data_quality.build_calendar_month_windows
    call_count = 0

    def differing(*args, **kwargs):
        nonlocal call_count
        result = list(original(*args, **kwargs))
        call_count += 1
        if call_count == 2:
            result[0] = replace(result[0], is_end=result[0].is_end - pd.Timedelta(minutes=30))
        return result

    monkeypatch.setattr(data_quality, "build_calendar_month_windows", differing)
    with pytest.raises(DataQualityError, match="boundaries differ"):
        build_authoritative_windows(
            spec,
            (_source_for_index(full), _source_for_index(full, "ZZZUSDT")),
        )


def test_authoritative_windows_handle_leap_february_and_variable_months():
    spec = load_run_spec(CURRENT_RUN_SPEC, validate_inventory=False)
    changed_generation = copy.deepcopy(spec.generation)
    changed_generation["windows"].update(
        {
            "requested_start": "2024-01-01",
            "requested_end": "2024-05-01",
            "is_period_months": 1,
            "oos_period_months": 1,
            "expected_window_count": 3,
        }
    )
    changed = replace(spec, generation=changed_generation)
    index = pd.date_range("2023-12-01", "2024-05-01 23:30", freq="30min", tz="UTC")
    windows = build_authoritative_windows(changed, (_source_for_index(index),))

    assert windows[0].is_end == pd.Timestamp("2024-01-31T23:30:00Z")
    assert windows[0].oos_end == pd.Timestamp("2024-02-29T23:30:00Z")
    assert windows[1].is_end == pd.Timestamp("2024-02-29T23:30:00Z")
    assert windows[2].oos_end == pd.Timestamp("2024-04-30T23:30:00Z")


def test_segment_preparation_is_independent_and_exact():
    spec = load_run_spec(CURRENT_RUN_SPEC, validate_inventory=False)
    index = pd.date_range("2025-08-01", "2026-08-01 23:30", freq="30min", tz="UTC")
    source = _source_for_index(index)
    window = build_authoritative_windows(spec, (source,))[0]
    prepared_is = prepare_segment(source, window, "is", warmup_bars=1000, timeframe_minutes=30)
    prepared_oos = prepare_segment(source, window, "oos", warmup_bars=1000, timeframe_minutes=30)

    assert prepared_is.trade_start_idx == prepared_oos.trade_start_idx == 1000
    assert prepared_is.evaluation_row_count == 2_928
    assert prepared_oos.evaluation_row_count == 1_488
    assert len(prepared_is.dataframe) == 3_928
    assert len(prepared_oos.dataframe) == 2_488
    assert prepared_is.dataframe is not prepared_oos.dataframe


def test_insufficient_warmup_is_rejected():
    spec = load_run_spec(CURRENT_RUN_SPEC, validate_inventory=False)
    index = pd.date_range("2025-09-25", "2026-08-01 23:30", freq="30min", tz="UTC")
    source = _source_for_index(index)
    window = build_authoritative_windows(spec, (source,))[0]
    with pytest.raises(DataQualityError, match="insufficient warmup"):
        prepare_segment(source, window, "is", warmup_bars=1000, timeframe_minutes=30)


def test_source_preservation_includes_bytes_size_and_mtime(tmp_path):
    data_root, entry, target = _small_pack(tmp_path)
    source = validate_source(data_root, entry, timeframe_minutes=30)
    before = (target.read_bytes(), target.stat().st_size, target.stat().st_mtime_ns)
    assert verify_source_preservation((source,))["sha256_size_mtime_unchanged"] is True
    assert (target.read_bytes(), target.stat().st_size, target.stat().st_mtime_ns) == before

    os.utime(target, ns=(target.stat().st_atime_ns, target.stat().st_mtime_ns + 1_000_000))
    with pytest.raises(DataQualityError, match="hash, size, or mtime changed"):
        verify_source_preservation((source,))
