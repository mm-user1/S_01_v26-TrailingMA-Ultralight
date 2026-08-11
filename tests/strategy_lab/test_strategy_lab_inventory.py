from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.strategy_lab.config import DATA_ROOT_ENV_VAR, canonical_sha256
from tools.strategy_lab.inventory import (
    EXPECTED_HEADER,
    InventoryError,
    build_inventory,
    calculate_size_steps,
    load_inventory,
    parse_filename,
    resolve_data_root,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CURRENT_INVENTORY = REPO_ROOT / "tools" / "strategy_lab" / "runspecs" / "tickers_current.json"


def _write_csv(root, exchange, symbol, closes=(100.0, 101.0), *, header=None, timeframe=30):
    path = root / f"{exchange}_{symbol}.P, {timeframe} 2025.08.01-2026.08.01.csv"
    rows = [",".join(EXPECTED_HEADER if header is None else header)]
    for index, close in enumerate(closes):
        rows.append(f"{1754006400 + index * 1800},100,102,99,{close},1")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _build(root, **overrides):
    kwargs = {
        "expected_ticker_count": 2,
        "development_ticker_count": 1,
        "expected_timeframe_minutes": 30,
        "initial_capital": 1000.0,
        "risk_per_trade_pct": 2.0,
        "contract_size": 0.0001,
        "max_stop_pct": 8.0,
        "minimum_size_steps": 100,
    }
    kwargs.update(overrides)
    return build_inventory(root, **kwargs)


def test_data_root_precedence_and_missing_contract(tmp_path):
    explicit = tmp_path / "explicit"
    environment = tmp_path / "environment"
    explicit.mkdir()
    environment.mkdir()

    assert resolve_data_root(explicit, environ={DATA_ROOT_ENV_VAR: str(environment)}) == explicit.resolve()
    assert resolve_data_root(None, environ={DATA_ROOT_ENV_VAR: str(environment)}) == environment.resolve()
    with pytest.raises(InventoryError, match="--data-root.*MERLIN_STRATEGY_LAB_DATA_ROOT"):
        resolve_data_root(None, environ={})


@pytest.mark.parametrize("exchange", ["OKX", "BYBIT"])
def test_filename_parser_records_half_open_current_shape(exchange):
    parsed = parse_filename(
        f"{exchange}_COREUSDT.P, 30 2025.08.01-2026.08.01.csv",
        expected_timeframe_minutes=30,
    )

    assert parsed.exchange == exchange
    assert parsed.raw_instrument_label == "COREUSDT.P"
    assert parsed.canonical_symbol == "COREUSDT"
    assert parsed.start == "2025-08-01T00:00:00Z"
    assert parsed.end == "2026-08-01T00:00:00Z"


@pytest.mark.parametrize(
    ("filename", "message"),
    [
        ("OKX_COREUSDT.csv", "does not match"),
        ("OKX_COREUSDT.P, 15 2025.08.01-2026.08.01.csv", "timeframe 15"),
        ("OKX_КОРUSDT.P, 30 2025.08.01-2026.08.01.csv", "non-ASCII"),
        ("OKX_COREUSDT.P, 30 2026.08.01-2025.08.01.csv", "non-increasing"),
    ],
)
def test_filename_parser_rejects_invalid_current_sources(filename, message):
    with pytest.raises(InventoryError, match=message):
        parse_filename(filename, expected_timeframe_minutes=30)


def test_inventory_is_deterministic_independent_of_filesystem_order(tmp_path, monkeypatch):
    _write_csv(tmp_path, "OKX", "AAAUSDT")
    _write_csv(tmp_path, "BYBIT", "ZZZUSDT")
    normal = _build(tmp_path)
    original = Path.iterdir

    def reverse_iterdir(path):
        values = list(original(path))
        return iter(reversed(values)) if path == tmp_path else iter(values)

    monkeypatch.setattr(Path, "iterdir", reverse_iterdir)
    reversed_result = _build(tmp_path)

    assert reversed_result == normal
    assert [entry["cell"] for entry in normal["entries"]] == ["dev", "holdout"]
    assert [entry["split_digest"] for entry in normal["entries"]] == sorted(
        entry["split_digest"] for entry in normal["entries"]
    )


def test_duplicate_canonical_symbols_are_rejected(tmp_path):
    _write_csv(tmp_path, "OKX", "COREUSDT")
    _write_csv(tmp_path, "BYBIT", "COREUSDT")

    with pytest.raises(InventoryError, match="duplicate canonical symbols"):
        _build(tmp_path)


def test_wrong_header_and_sizing_failure_are_rejected(tmp_path):
    _write_csv(tmp_path, "OKX", "AAAUSDT", header=["bad"])
    _write_csv(tmp_path, "BYBIT", "BBBUSD")
    with pytest.raises(InventoryError, match="header must be"):
        _build(tmp_path)

    for item in tmp_path.iterdir():
        item.unlink()
    _write_csv(tmp_path, "OKX", "AAAUSDT", closes=(25200.0,))
    _write_csv(tmp_path, "BYBIT", "BBBUSD", closes=(100.0,))
    with pytest.raises(InventoryError, match="sizing headroom 99 steps"):
        _build(tmp_path)


def test_sizing_headroom_exact_pass_fail_boundary():
    common = {
        "initial_capital": 1000.0,
        "risk_per_trade_pct": 2.0,
        "contract_size": 0.0001,
        "max_stop_pct": 8.0,
    }

    assert calculate_size_steps(max_close=25000.0, **common) == 100
    assert calculate_size_steps(max_close=25200.0, **common) == 99


def test_source_files_are_never_opened_for_write(tmp_path, monkeypatch):
    sources = {
        _write_csv(tmp_path, "OKX", "AAAUSDT").resolve(),
        _write_csv(tmp_path, "BYBIT", "BBBUSD").resolve(),
    }
    original = Path.open
    seen_modes = []

    def guarded_open(path, mode="r", *args, **kwargs):
        if path.resolve() in sources:
            seen_modes.append(mode)
            assert not any(flag in mode for flag in ("w", "a", "+", "x"))
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    _build(tmp_path)

    assert seen_modes and set(seen_modes) == {"r", "rb"}


def test_source_symlinks_are_rejected_before_read(tmp_path, monkeypatch):
    first = _write_csv(tmp_path, "OKX", "AAAUSDT")
    _write_csv(tmp_path, "BYBIT", "BBBUSD")
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == first or original(path))

    with pytest.raises(InventoryError, match="source symlinks"):
        _build(tmp_path)


def test_frozen_assignment_is_authoritative_on_load(tmp_path):
    _write_csv(tmp_path, "OKX", "AAAUSDT")
    _write_csv(tmp_path, "BYBIT", "BBBUSD")
    inventory = _build(tmp_path)
    inventory["entries"][0]["cell"] = "holdout"
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")

    with pytest.raises(InventoryError, match="frozen assignment mismatch"):
        load_inventory(path)


def test_current_inventory_has_exact_stage_b_shape_and_no_host_facts():
    inventory = load_inventory(CURRENT_INVENTORY)
    forbidden = {"absolute_root", "resolved_absolute_root", "host", "platform", "mtime", "generation_time", "verification_timestamp"}

    def keys(value):
        if isinstance(value, dict):
            for name, child in value.items():
                yield name
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    assert len(inventory.entries) == 118
    assert inventory.development_count == 24
    assert inventory.holdout_count == 94
    assert sum(entry["exchange"] == "OKX" for entry in inventory.entries) == 110
    assert sum(entry["exchange"] == "BYBIT" for entry in inventory.entries) == 8
    assert len({entry["canonical_symbol"] for entry in inventory.entries}) == 118
    assert min(entry["size_steps"] for entry in inventory.entries) == 405
    assert forbidden.isdisjoint(keys(inventory.raw))
    assert inventory.sha256 == canonical_sha256(inventory.raw)
