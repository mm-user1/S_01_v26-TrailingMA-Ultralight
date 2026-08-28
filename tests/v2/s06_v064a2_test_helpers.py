from __future__ import annotations

import csv
import json
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import pandas as pd

from core.backtest_engine import load_data, prepare_dataset_with_warmup
from core.engine_v2.compiled_kernel import evaluate_compiled_batch
from core.engine_v2.runner import run_v2_strategy
from strategies.s06_r_trend_v06_4_a2_b2 import strategy as strategy_module


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = REPO_ROOT / "data" / "baseline_v2" / "s06_r_trend_v06_4_a2"
MARKET_DATA_PATH = REPO_ROOT / "data" / "raw" / "OKX_SUIUSDT.P, 30 2025.01.01-2026.02.01.csv"
BASELINE_START = pd.Timestamp("2025-08-01T00:00:00Z")
BASELINE_END = pd.Timestamp("2025-12-01T00:00:00Z")
REFERENCE_IDS = (
    "reference_a_reversal_r_trail",
    "reference_b_trend_r_trail",
    "reference_c_reversal_chandelier",
    "reference_d_trend_chandelier",
    "reference_e_reversal_fixed_af_sar",
    "reference_f_trend_fixed_af_sar",
)
RUNTIME_PARAMS = {
    "dateFilter": True,
    "start": "2025-08-01T00:00:00Z",
    "end": "2025-12-01T00:00:00Z",
}


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_reference(reference_id: str) -> tuple[dict, dict, list[dict[str, str]]]:
    root = BASELINE_ROOT / reference_id
    params = json.loads((root / "params.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "tradingview_summary.json").read_text(encoding="utf-8"))
    return params, summary, csv_rows(root / "trades_normalized_utc.csv")


def merged_reference_params(reference_id: str, extra: dict | None = None) -> dict:
    params, _, _ = load_reference(reference_id)
    merged = strategy_module.normalized_params({**params["strategy_inputs"], **RUNTIME_PARAMS})
    merged.update(extra or {})
    return merged


@lru_cache(maxsize=1)
def prepared_reference_dataset():
    return prepare_dataset_with_warmup(
        load_data(MARKET_DATA_PATH),
        BASELINE_START,
        BASELINE_END,
        1000,
    )


@lru_cache(maxsize=None)
def run_reference(reference_id: str, *, compute_mtm: bool = True):
    params = merged_reference_params(reference_id)
    prepared, trade_start_idx = prepared_reference_dataset()
    data = strategy_module.build_v2_execution_data(prepared, params)
    return run_v2_strategy(
        data=data,
        profile=strategy_module.load_profile(),
        params=params,
        trade_start_idx=trade_start_idx,
        compute_max_drawdown_mtm=compute_mtm,
    )


@lru_cache(maxsize=None)
def run_compiled(reference_id: str, *, compute_mtm: bool = True):
    params = merged_reference_params(reference_id)
    prepared, trade_start_idx = prepared_reference_dataset()
    data = strategy_module.build_v2_execution_data(prepared, params)
    return evaluate_compiled_batch(
        data=data,
        profile=strategy_module.load_profile(),
        params_batch=[params],
        trade_start_idx=trade_start_idx,
        n_workers=1,
        compute_max_drawdown_mtm=compute_mtm,
    )


def iso_timestamp(value) -> str:
    return pd.Timestamp(value).isoformat().replace("+00:00", "Z")


def tradingview_comparison(reference_id: str) -> dict:
    _, _, rows = load_reference(reference_id)
    run = run_reference(reference_id)
    result = run.strategy_result
    fields = {
        "direction": lambda row, trade: (row["direction"], trade.direction),
        "entry_time": lambda row, trade: (row["entry_time_utc"], iso_timestamp(trade.entry_time)),
        "exit_time": lambda row, trade: (row["exit_time_utc"], iso_timestamp(trade.exit_time)),
    }
    exact = {}
    for name, getter in fields.items():
        mismatches = []
        for index, (row, trade) in enumerate(zip(rows, result.trades), start=1):
            expected, actual = getter(row, trade)
            if expected != actual:
                mismatches.append((index, expected, actual))
        exact[name] = {"count": len(mismatches), "first": mismatches[0] if mismatches else None}

    numeric_fields = {
        "entry_price": ("entry_price_usdt", "entry_price"),
        "exit_price": ("exit_price_usdt", "exit_price"),
        "quantity": ("size_qty", "size"),
        "net_pnl": ("net_pnl_usdt", "net_pnl"),
    }
    numeric = {}
    for name, (column, attribute) in numeric_fields.items():
        deltas = [
            abs(float(row[column]) - float(getattr(trade, attribute)))
            for row, trade in zip(rows, result.trades)
        ]
        numeric[name] = {
            "nonzero": sum(delta > 1e-12 for delta in deltas),
            "max_abs": max(deltas, default=0.0),
        }
    return {
        "reference": reference_id,
        "tradingview_count": len(rows),
        "merlin_count": result.total_trades,
        "exact": exact,
        "numeric": numeric,
        "merlin_net_profit_pct": result.net_profit_pct,
        "merlin_max_drawdown_pct": result.max_drawdown_pct,
        "merlin_max_drawdown_mtm_pct": run.max_drawdown_mtm_pct,
    }


def entry_alignment(reference_id: str):
    _, _, rows = load_reference(reference_id)
    trades = run_reference(reference_id).strategy_result.trades
    expected = [(row["direction"], row["entry_time_utc"]) for row in rows]
    actual = [(trade.direction, iso_timestamp(trade.entry_time)) for trade in trades]
    return SequenceMatcher(a=expected, b=actual, autojunk=False).get_opcodes()


def trade_context(reference_id: str, start: int, stop: int) -> dict:
    _, _, rows = load_reference(reference_id)
    trades = run_reference(reference_id).strategy_result.trades
    return {
        "tradingview": [
            {
                "number": index + 1,
                "direction": row["direction"],
                "entry": row["entry_time_utc"],
                "exit": row["exit_time_utc"],
                "entry_price": float(row["entry_price_usdt"]),
                "exit_price": float(row["exit_price_usdt"]),
            }
            for index, row in enumerate(rows[start:stop], start=start)
        ],
        "merlin": [
            {
                "number": index + 1,
                "direction": trade.direction,
                "entry": iso_timestamp(trade.entry_time),
                "exit": iso_timestamp(trade.exit_time),
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
            }
            for index, trade in enumerate(trades[start:stop], start=start)
        ],
    }
