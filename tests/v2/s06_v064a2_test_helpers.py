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
    trades = run.strategy_result.trades
    expected_identities = [(row["direction"], row["entry_time_utc"]) for row in rows]
    actual_identities = [(trade.direction, iso_timestamp(trade.entry_time)) for trade in trades]
    opcodes = SequenceMatcher(
        a=expected_identities,
        b=actual_identities,
        autojunk=False,
    ).get_opcodes()
    matched_indices = [
        (expected_index, actual_index)
        for tag, expected_start, expected_stop, actual_start, actual_stop in opcodes
        if tag == "equal"
        for expected_index, actual_index in zip(
            range(expected_start, expected_stop),
            range(actual_start, actual_stop),
        )
    ]
    unmatched_tradingview = [
        {"number": index + 1, "identity": expected_identities[index]}
        for tag, expected_start, expected_stop, _, _ in opcodes
        if tag in {"delete", "replace"}
        for index in range(expected_start, expected_stop)
    ]
    unmatched_merlin = [
        {"number": index + 1, "identity": actual_identities[index]}
        for tag, _, _, actual_start, actual_stop in opcodes
        if tag in {"insert", "replace"}
        for index in range(actual_start, actual_stop)
    ]
    fields = {
        "direction": lambda row, trade: (row["direction"], trade.direction),
        "entry_time": lambda row, trade: (row["entry_time_utc"], iso_timestamp(trade.entry_time)),
        "exit_time": lambda row, trade: (row["exit_time_utc"], iso_timestamp(trade.exit_time)),
    }
    exact = {}
    for name, getter in fields.items():
        mismatches = []
        for expected_index, actual_index in matched_indices:
            row, trade = rows[expected_index], trades[actual_index]
            expected, actual = getter(row, trade)
            if expected != actual:
                mismatches.append(
                    {
                        "tradingview_number": expected_index + 1,
                        "merlin_number": actual_index + 1,
                        "expected": expected,
                        "actual": actual,
                    }
                )
        exact[name] = {"count": len(mismatches), "mismatches": mismatches}

    numeric_fields = {
        "entry_price": ("entry_price_usdt", "entry_price"),
        "exit_price": ("exit_price_usdt", "exit_price"),
        "quantity": ("size_qty", "size"),
        "net_pnl": ("net_pnl_usdt", "net_pnl"),
    }
    numeric = {}
    for name, (column, attribute) in numeric_fields.items():
        deltas = []
        for expected_index, actual_index in matched_indices:
            delta = abs(
                float(rows[expected_index][column])
                - float(getattr(trades[actual_index], attribute))
            )
            deltas.append((expected_index + 1, actual_index + 1, delta))
        maximum = max(deltas, key=lambda item: item[2], default=None)
        numeric[name] = {
            "nonzero": sum(delta > 1e-12 for _, _, delta in deltas),
            "max_abs": maximum[2] if maximum else 0.0,
            "max_at": maximum[:2] if maximum else None,
            "above_csv_rounding": [
                (expected_number, actual_number)
                for expected_number, actual_number, delta in deltas
                if delta > {"entry_price": 0.0001, "exit_price": 0.0001, "quantity": 0.01000000000001, "net_pnl": 0.007}[name]
            ],
        }
    return {
        "reference": reference_id,
        "tradingview_count": len(rows),
        "merlin_count": len(trades),
        "alignment": opcodes,
        "matched_count": len(matched_indices),
        "unmatched_tradingview": unmatched_tradingview,
        "unmatched_merlin": unmatched_merlin,
        "exact": exact,
        "numeric": numeric,
        "merlin_net_profit_pct": run.strategy_result.net_profit_pct,
        "merlin_max_drawdown_pct": run.strategy_result.max_drawdown_pct,
        "merlin_max_drawdown_mtm_pct": run.max_drawdown_mtm_pct,
    }


def entry_alignment(reference_id: str):
    return tradingview_comparison(reference_id)["alignment"]


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
