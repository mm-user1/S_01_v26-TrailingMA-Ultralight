"""Deterministic, read-only Strategy Lab market-data inventory builder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import (
    DATA_ROOT_ENV_VAR,
    INVENTORY_SCHEMA_VERSION,
    StrategyLabConfigError,
    canonical_json_bytes,
    canonical_sha256,
)


EXPECTED_HEADER = ["time", "open", "high", "low", "close", "Volume"]
SPLIT_ALGORITHM = "sha256_canonical_symbol_ascending_v1"
FILENAME_RE = re.compile(
    r"^(?P<exchange>[A-Za-z0-9]+)_(?P<symbol>[A-Za-z0-9]+)\.P, "
    r"(?P<timeframe>[0-9]+) (?P<start>[0-9]{4}\.[0-9]{2}\.[0-9]{2})-"
    r"(?P<end>[0-9]{4}\.[0-9]{2}\.[0-9]{2})\.csv$"
)


class InventoryError(StrategyLabConfigError):
    """Concise source/inventory validation failure."""


@dataclass(frozen=True)
class ParsedFilename:
    exchange: str
    raw_instrument_label: str
    canonical_symbol: str
    timeframe_minutes: int
    start: str
    end: str


@dataclass(frozen=True)
class Inventory:
    source_path: Path
    raw: Mapping[str, Any]
    entries: tuple[Mapping[str, Any], ...]
    sha256: str
    ticker_list_digest: str
    development_count: int
    holdout_count: int


def resolve_data_root(
    explicit: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve only the explicit root or the Strategy Lab-specific environment value."""

    env = os.environ if environ is None else environ
    raw = explicit if explicit is not None else env.get(DATA_ROOT_ENV_VAR)
    if raw is None or not str(raw).strip():
        raise InventoryError(
            f"data root is required; pass --data-root or set {DATA_ROOT_ENV_VAR}."
        )
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise InventoryError(f"data root does not exist or is not a directory: {root}")
    return root


def _iso_date(raw: str, field: str) -> str:
    try:
        parsed = datetime.strptime(raw, "%Y.%m.%d").replace(tzinfo=UTC)
    except ValueError:
        raise InventoryError(f"{field}: invalid filename date {raw!r}.") from None
    return parsed.isoformat().replace("+00:00", "Z")


def parse_filename(filename: str, *, expected_timeframe_minutes: int) -> ParsedFilename:
    if not filename.isascii():
        raise InventoryError(f"filename: {filename!r} contains a non-ASCII identifier.")
    match = FILENAME_RE.fullmatch(filename)
    if match is None:
        raise InventoryError(
            f"filename: {filename!r} does not match '<EXCHANGE>_<SYMBOL>.P, 30 YYYY.MM.DD-YYYY.MM.DD.csv'."
        )
    exchange_raw = match.group("exchange")
    symbol_raw = match.group("symbol")
    try:
        exchange_raw.encode("ascii")
        symbol_raw.encode("ascii")
    except UnicodeEncodeError:
        raise InventoryError(f"filename: {filename!r} contains a non-ASCII identifier.") from None
    timeframe = int(match.group("timeframe"))
    if timeframe != expected_timeframe_minutes:
        raise InventoryError(
            f"filename: {filename!r} declares timeframe {timeframe}, expected {expected_timeframe_minutes}."
        )
    start = _iso_date(match.group("start"), "filename start")
    end = _iso_date(match.group("end"), "filename end")
    if start >= end:
        raise InventoryError(f"filename: {filename!r} has a non-increasing half-open date interval.")
    canonical_symbol = symbol_raw.upper()
    return ParsedFilename(
        exchange=exchange_raw.upper(),
        raw_instrument_label=f"{symbol_raw}.P",
        canonical_symbol=canonical_symbol,
        timeframe_minutes=timeframe,
        start=start,
        end=end,
    )


def normalize_timestamp(raw: str, field: str) -> str:
    value = raw.strip()
    try:
        if re.fullmatch(r"[+-]?[0-9]+(?:\.[0-9]+)?", value):
            number = float(value)
            if not math.isfinite(number):
                raise ValueError
            if abs(number) >= 100_000_000_000:
                number /= 1000.0
            parsed = datetime.fromtimestamp(number, tz=UTC)
        else:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            else:
                parsed = parsed.astimezone(UTC)
    except (ValueError, OverflowError, OSError):
        raise InventoryError(f"{field}: invalid timestamp {raw!r}.") from None
    return parsed.isoformat().replace("+00:00", "Z")


def ticker_list_digest(symbols: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for symbol in symbols:
        normalized = symbol.upper()
        try:
            encoded = normalized.encode("ascii")
        except UnicodeEncodeError:
            raise InventoryError(f"canonical symbol {symbol!r} is not ASCII.") from None
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def calculate_size_steps(
    *,
    max_close: float,
    initial_capital: float,
    risk_per_trade_pct: float,
    contract_size: float,
    max_stop_pct: float,
) -> int:
    values = {
        "max_close": max_close,
        "initial_capital": initial_capital,
        "risk_per_trade_pct": risk_per_trade_pct,
        "contract_size": contract_size,
        "max_stop_pct": max_stop_pct,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InventoryError(f"sizing.{name}: expected a finite positive number.")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise InventoryError(f"sizing.{name}: expected a finite positive number.")
    risk_cash = float(initial_capital) * float(risk_per_trade_pct) / 100.0
    worst_stop_distance = float(max_close) * float(max_stop_pct) / 100.0
    raw_size = risk_cash / worst_stop_distance
    return math.floor(raw_size / float(contract_size))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_source(
    path: Path,
    *,
    parsed: ParsedFilename,
    initial_capital: float,
    risk_per_trade_pct: float,
    contract_size: float,
    max_stop_pct: float,
    minimum_size_steps: int,
) -> dict[str, Any]:
    if path.is_symlink():
        raise InventoryError(f"source file symlinks are not allowed: {path.name}")
    row_count = 0
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    max_close = -math.inf
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header != EXPECTED_HEADER:
                raise InventoryError(
                    f"source {path.name}: header must be {','.join(EXPECTED_HEADER)}."
                )
            for line_number, row in enumerate(reader, start=2):
                if len(row) != len(EXPECTED_HEADER):
                    raise InventoryError(
                        f"source {path.name}:{line_number}: expected {len(EXPECTED_HEADER)} columns."
                    )
                timestamp = normalize_timestamp(row[0], f"source {path.name}:{line_number}.time")
                try:
                    close = float(row[4])
                except ValueError:
                    raise InventoryError(
                        f"source {path.name}:{line_number}.close: expected a number."
                    ) from None
                if math.isfinite(close) and close > 0.0:
                    max_close = max(max_close, close)
                if first_timestamp is None:
                    first_timestamp = timestamp
                last_timestamp = timestamp
                row_count += 1
    except UnicodeError as exc:
        raise InventoryError(f"source {path.name}: invalid UTF-8: {exc}") from None
    if row_count == 0 or first_timestamp is None or last_timestamp is None:
        raise InventoryError(f"source {path.name}: no data rows.")
    if not math.isfinite(max_close):
        raise InventoryError(f"source {path.name}: no finite positive Close value.")
    size_steps = calculate_size_steps(
        max_close=max_close,
        initial_capital=initial_capital,
        risk_per_trade_pct=risk_per_trade_pct,
        contract_size=contract_size,
        max_stop_pct=max_stop_pct,
    )
    if size_steps < minimum_size_steps:
        raise InventoryError(
            f"source {path.name}: sizing headroom {size_steps} steps is below required {minimum_size_steps}."
        )
    return {
        "relative_filename": path.name,
        "exchange": parsed.exchange,
        "raw_instrument_label": parsed.raw_instrument_label,
        "canonical_symbol": parsed.canonical_symbol,
        "timeframe_minutes": parsed.timeframe_minutes,
        "filename_start": parsed.start,
        "filename_end": parsed.end,
        "header_sha256": canonical_sha256(EXPECTED_HEADER),
        "row_count": row_count,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "max_close": max_close,
        "size_steps": size_steps,
        "file_size": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def build_inventory(
    data_root: str | Path,
    *,
    expected_ticker_count: int,
    development_ticker_count: int,
    expected_timeframe_minutes: int,
    initial_capital: float,
    risk_per_trade_pct: float,
    contract_size: float,
    max_stop_pct: float,
    minimum_size_steps: int = 100,
) -> dict[str, Any]:
    """Build canonical inventory facts without writing to or beside source files."""

    root = resolve_data_root(data_root)
    if (
        isinstance(expected_ticker_count, bool)
        or not isinstance(expected_ticker_count, int)
        or expected_ticker_count <= 1
    ):
        raise InventoryError("expected_ticker_count: expected an integer > 1.")
    if (
        isinstance(development_ticker_count, bool)
        or not isinstance(development_ticker_count, int)
        or development_ticker_count <= 0
        or development_ticker_count >= expected_ticker_count
    ):
        raise InventoryError("development_ticker_count: must leave non-empty dev and holdout cells.")
    entries_on_disk = list(root.iterdir())
    symlinks = sorted(item.name for item in entries_on_disk if item.is_symlink())
    if symlinks:
        raise InventoryError(f"data root contains source symlinks: {', '.join(symlinks)}.")
    directories = sorted(item.name for item in entries_on_disk if item.is_dir())
    if directories:
        raise InventoryError(f"data root contains subdirectories: {', '.join(directories)}.")
    files = [item for item in entries_on_disk if item.is_file() or item.is_symlink()]
    if len(files) != expected_ticker_count:
        raise InventoryError(
            f"data root: expected {expected_ticker_count} files, found {len(files)}."
        )
    records: list[dict[str, Any]] = []
    filename_interval: tuple[str, str] | None = None
    for path in sorted(files, key=lambda item: item.name):
        parsed = parse_filename(path.name, expected_timeframe_minutes=expected_timeframe_minutes)
        interval = (parsed.start, parsed.end)
        if filename_interval is None:
            filename_interval = interval
        elif interval != filename_interval:
            raise InventoryError(f"source {path.name}: filename interval differs from the pack.")
        record = _read_source(
            path,
            parsed=parsed,
            initial_capital=initial_capital,
            risk_per_trade_pct=risk_per_trade_pct,
            contract_size=contract_size,
            max_stop_pct=max_stop_pct,
            minimum_size_steps=minimum_size_steps,
        )
        record["split_digest"] = hashlib.sha256(parsed.canonical_symbol.encode("ascii")).hexdigest()
        records.append(record)
    symbols = [record["canonical_symbol"] for record in records]
    duplicates = sorted(symbol for symbol in set(symbols) if symbols.count(symbol) > 1)
    if duplicates:
        raise InventoryError(f"duplicate canonical symbols: {', '.join(duplicates)}.")
    records.sort(key=lambda record: (record["split_digest"], record["canonical_symbol"]))
    for index, record in enumerate(records):
        record["cell"] = "dev" if index < development_ticker_count else "holdout"
    if filename_interval is None:
        raise InventoryError("data root: no source files.")
    ordered_symbols = [record["canonical_symbol"] for record in records]
    return {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "header": list(EXPECTED_HEADER),
        "timeframe_minutes": expected_timeframe_minutes,
        "filename_interval": {
            "start": filename_interval[0],
            "end": filename_interval[1],
            "boundary": "half_open",
        },
        "split": {
            "algorithm": SPLIT_ALGORITHM,
            "development_ticker_count": development_ticker_count,
            "holdout_ticker_count": expected_ticker_count - development_ticker_count,
        },
        "ticker_list_digest": ticker_list_digest(ordered_symbols),
        "entries": records,
    }


def _require_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InventoryError(f"{field}: expected an integer >= {minimum} (booleans are invalid).")
    return value


def _require_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise InventoryError(f"{field}: expected a lowercase SHA-256 hex digest.")
    return value


def load_inventory(path: str | Path) -> Inventory:
    source = Path(path).resolve()
    try:
        with source.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise InventoryError(f"inventory: malformed JSON at line {exc.lineno}, column {exc.colno}.") from None
    except OSError as exc:
        raise InventoryError(f"inventory: cannot read {source}: {exc}") from None
    if not isinstance(raw, Mapping):
        raise InventoryError("inventory: expected an object.")
    expected_top = {"schema_version", "header", "timeframe_minutes", "filename_interval", "split", "ticker_list_digest", "entries"}
    if set(raw) != expected_top:
        raise InventoryError("inventory: fields do not match strategy_lab_inventory_v1.")
    if raw["schema_version"] != INVENTORY_SCHEMA_VERSION:
        raise InventoryError(f"inventory.schema_version: expected '{INVENTORY_SCHEMA_VERSION}'.")
    if raw["header"] != EXPECTED_HEADER:
        raise InventoryError("inventory.header: unsupported schema.")
    timeframe = _require_int(raw["timeframe_minutes"], "inventory.timeframe_minutes", minimum=1)
    interval = raw["filename_interval"]
    if not isinstance(interval, Mapping) or set(interval) != {"start", "end", "boundary"} or interval["boundary"] != "half_open":
        raise InventoryError("inventory.filename_interval: invalid half-open interval contract.")
    if not isinstance(interval["start"], str) or not isinstance(interval["end"], str) or interval["start"] >= interval["end"]:
        raise InventoryError("inventory.filename_interval: dates must be increasing UTC strings.")
    split = raw["split"]
    if not isinstance(split, Mapping) or set(split) != {"algorithm", "development_ticker_count", "holdout_ticker_count"}:
        raise InventoryError("inventory.split: invalid split contract.")
    if split["algorithm"] != SPLIT_ALGORITHM:
        raise InventoryError("inventory.split.algorithm: unsupported algorithm.")
    dev_count = _require_int(split["development_ticker_count"], "inventory.split.development_ticker_count", minimum=1)
    holdout_count = _require_int(split["holdout_ticker_count"], "inventory.split.holdout_ticker_count", minimum=1)
    entries = raw["entries"]
    if not isinstance(entries, list) or len(entries) != dev_count + holdout_count:
        raise InventoryError("inventory.entries: count does not match split totals.")
    entry_keys = {
        "relative_filename", "exchange", "raw_instrument_label", "canonical_symbol",
        "timeframe_minutes", "filename_start", "filename_end", "header_sha256",
        "row_count", "first_timestamp", "last_timestamp", "max_close", "size_steps",
        "file_size", "sha256", "split_digest", "cell",
    }
    symbols: list[str] = []
    prior_key: tuple[str, str] | None = None
    normalized_entries: list[Mapping[str, Any]] = []
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, Mapping) or set(raw_entry) != entry_keys:
            raise InventoryError(f"inventory.entries[{index}]: fields do not match the inventory contract.")
        filename = raw_entry["relative_filename"]
        if not isinstance(filename, str) or Path(filename).name != filename or Path(filename).is_absolute():
            raise InventoryError(f"inventory.entries[{index}].relative_filename: expected a relative filename.")
        symbol = raw_entry["canonical_symbol"]
        if not isinstance(symbol, str) or not symbol or symbol != symbol.upper() or not symbol.isascii() or not symbol.isalnum():
            raise InventoryError(f"inventory.entries[{index}].canonical_symbol: expected uppercase ASCII letters/digits.")
        split_digest = _require_sha(raw_entry["split_digest"], f"inventory.entries[{index}].split_digest")
        if split_digest != hashlib.sha256(symbol.encode("ascii")).hexdigest():
            raise InventoryError(f"inventory.entries[{index}].split_digest: does not match canonical symbol.")
        sort_key = (split_digest, symbol)
        if prior_key is not None and sort_key <= prior_key:
            raise InventoryError("inventory.entries: persisted split ordering is not canonical.")
        prior_key = sort_key
        expected_cell = "dev" if index < dev_count else "holdout"
        if raw_entry["cell"] != expected_cell:
            raise InventoryError(f"inventory.entries[{index}].cell: frozen assignment mismatch.")
        if raw_entry["timeframe_minutes"] != timeframe:
            raise InventoryError(f"inventory.entries[{index}].timeframe_minutes: pack mismatch.")
        if raw_entry["filename_start"] != interval["start"] or raw_entry["filename_end"] != interval["end"]:
            raise InventoryError(f"inventory.entries[{index}]: filename interval differs from the pack.")
        if raw_entry["header_sha256"] != canonical_sha256(EXPECTED_HEADER):
            raise InventoryError(f"inventory.entries[{index}].header_sha256: header digest mismatch.")
        _require_int(raw_entry["row_count"], f"inventory.entries[{index}].row_count", minimum=1)
        _require_int(raw_entry["size_steps"], f"inventory.entries[{index}].size_steps", minimum=100)
        _require_int(raw_entry["file_size"], f"inventory.entries[{index}].file_size", minimum=1)
        max_close = raw_entry["max_close"]
        if isinstance(max_close, bool) or not isinstance(max_close, (int, float)) or not math.isfinite(float(max_close)) or float(max_close) <= 0.0:
            raise InventoryError(f"inventory.entries[{index}].max_close: expected a finite positive number.")
        _require_sha(raw_entry["sha256"], f"inventory.entries[{index}].sha256")
        _require_sha(raw_entry["header_sha256"], f"inventory.entries[{index}].header_sha256")
        symbols.append(symbol)
        normalized_entries.append(raw_entry)
    if len(set(symbols)) != len(symbols):
        raise InventoryError("inventory.entries: duplicate canonical symbols.")
    expected_ticker_digest = ticker_list_digest(symbols)
    if raw["ticker_list_digest"] != expected_ticker_digest:
        raise InventoryError("inventory.ticker_list_digest: ordered symbol digest mismatch.")
    return Inventory(
        source_path=source,
        raw=raw,
        entries=tuple(normalized_entries),
        sha256=canonical_sha256(raw),
        ticker_list_digest=expected_ticker_digest,
        development_count=dev_count,
        holdout_count=holdout_count,
    )


def write_canonical_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    output.write_text(payload, encoding="utf-8", newline="\n")


def build_local_provenance(data_root: Path, inventory: Mapping[str, Any], *, repo_root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    mtimes = {
        entry["relative_filename"]: (data_root / entry["relative_filename"]).stat().st_mtime_ns
        for entry in inventory["entries"]
    }
    return {
        "schema_version": "strategy_lab_local_provenance_v1",
        "resolved_absolute_root": str(data_root),
        "host": platform.node(),
        "platform": platform.platform(),
        "source_mtime_ns": mtimes,
        "verification_timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "tool_source_git_revision": revision,
        "inventory_sha256": canonical_sha256(inventory),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a read-only Strategy Lab Phase 0 inventory.")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path)
    parser.add_argument("--expected-ticker-count", type=int, required=True)
    parser.add_argument("--development-ticker-count", type=int, required=True)
    parser.add_argument("--timeframe-minutes", type=int, required=True)
    parser.add_argument("--initial-capital", type=float, required=True)
    parser.add_argument("--risk-per-trade-pct", type=float, required=True)
    parser.add_argument("--contract-size", type=float, required=True)
    parser.add_argument("--max-stop-pct", type=float, required=True)
    parser.add_argument("--minimum-size-steps", type=int, default=100)
    args = parser.parse_args(argv)
    try:
        root = resolve_data_root(args.data_root)
        inventory = build_inventory(
            root,
            expected_ticker_count=args.expected_ticker_count,
            development_ticker_count=args.development_ticker_count,
            expected_timeframe_minutes=args.timeframe_minutes,
            initial_capital=args.initial_capital,
            risk_per_trade_pct=args.risk_per_trade_pct,
            contract_size=args.contract_size,
            max_stop_pct=args.max_stop_pct,
            minimum_size_steps=args.minimum_size_steps,
        )
        write_canonical_json(args.output, inventory)
        if args.provenance_output is not None:
            write_canonical_json(
                args.provenance_output,
                build_local_provenance(root, inventory, repo_root=Path(__file__).resolve().parents[2]),
            )
    except InventoryError as exc:
        parser.exit(2, f"Strategy Lab inventory error: {exc}\n")
    print(
        json.dumps(
            {
                "inventory_sha256": canonical_sha256(inventory),
                "ticker_list_digest": inventory["ticker_list_digest"],
                "ticker_count": len(inventory["entries"]),
                "minimum_size_steps": min(entry["size_steps"] for entry in inventory["entries"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
