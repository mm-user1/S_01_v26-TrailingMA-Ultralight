from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from core.grid_v2 import GridV2Plan, GridV2ResultRow
from tools.strategy_lab.config import canonical_sha256
from tools.strategy_lab.inventory import EXPECTED_HEADER, build_inventory


REPO_ROOT = Path(__file__).resolve().parents[2]
CURRENT_RUN_SPEC = REPO_ROOT / "tools" / "strategy_lab" / "runspecs" / "s06_bracket_mvp.json"


def write_source(
    root: Path,
    exchange: str,
    symbol: str,
    *,
    start_seconds: int = 1_754_006_400,
    rows: int = 48,
    filename_end: str = "2025.08.01",
) -> Path:
    path = root / f"{exchange}_{symbol}.P, 30 2025.08.01-{filename_end}.csv"
    lines = [",".join(EXPECTED_HEADER)]
    for index in range(rows):
        middle = 100.0 + 0.001 * index + 1.5 * math.sin(index / 13.0)
        close = middle + 0.2 * math.sin(index / 5.0)
        high = max(middle, close) + 0.5
        low = min(middle, close) - 0.5
        lines.append(
            f"{start_seconds + index * 1800},{middle:.6f},{high:.6f},"
            f"{low:.6f},{close:.6f},100"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def build_small_inventory(root: Path) -> dict[str, Any]:
    return build_inventory(
        root,
        expected_ticker_count=2,
        development_ticker_count=1,
        expected_timeframe_minutes=30,
        initial_capital=1000.0,
        risk_per_trade_pct=2.0,
        contract_size=0.0001,
        max_stop_pct=8.0,
        minimum_size_steps=100,
    )


def write_full_synthetic_pack(root: Path) -> tuple[Path, dict[str, Any]]:
    data_root = root / "market"
    data_root.mkdir(parents=True)
    row_count = 17_568
    write_source(data_root, "OKX", "AAAUSDT", rows=row_count, filename_end="2026.08.01")
    write_source(data_root, "BYBIT", "ZZZUSDT", rows=row_count, filename_end="2026.08.01")
    inventory = build_small_inventory(data_root)
    inventory_path = root / "inventory.json"
    inventory_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    run_spec = copy.deepcopy(json.loads(CURRENT_RUN_SPEC.read_text(encoding="utf-8")))
    generation = run_spec["generation"]
    generation["inventory"] = {
        "inventory_path": "inventory.json",
        "inventory_sha256": canonical_sha256(inventory),
        "ticker_list_digest": inventory["ticker_list_digest"],
        "expected_ticker_count": 2,
    }
    exchange_counts: dict[str, int] = {}
    for entry in inventory["entries"]:
        exchange = entry["exchange"]
        exchange_counts[exchange] = exchange_counts.get(exchange, 0) + 1
    generation["market_data"]["expected_exchange_counts"] = exchange_counts
    run_spec["preregistration"]["split"]["development_ticker_count"] = 1
    run_spec["preregistration"]["split"]["holdout_ticker_count"] = 1
    run_spec_path = root / "run_spec.json"
    run_spec_path.write_text(
        json.dumps(run_spec, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return run_spec_path, inventory


def refresh_entry_file_facts(entry: dict[str, Any], path: Path) -> None:
    data = path.read_bytes()
    entry["file_size"] = len(data)
    entry["sha256"] = hashlib.sha256(data).hexdigest()


def fake_rows(plan: GridV2Plan, *, daily_none: bool = False) -> tuple[GridV2ResultRow, ...]:
    rows: list[GridV2ResultRow] = []
    for index in range(plan.deduped_candidate_count):
        candidate_id = index + 1
        rows.append(
            GridV2ResultRow(
                candidate_id=candidate_id,
                semantic_key=plan.candidate_table.semantic_key_for_index(index),
                canonical_identity=plan.candidate_table.canonical_identity_for_index(index),
                variant_name=plan.candidate_table.variant_name_for_index(index),
                grid_mode_name=plan.candidate_table.grid_mode_name_for_index(index),
                modes=plan.candidate_table.modes_for_index(index),
                params=plan.candidate_table.params_for_index(index),
                net_profit_pct=float(index) / 10.0,
                max_drawdown_pct=-float(index) / 100.0,
                romad=float("nan") if index == 0 else 1.0,
                profit_factor=float("inf") if index == 1 else 1.1,
                win_rate_pct=0.0,
                total_trades=0 if index == 0 else 1,
                winning_trades=0,
                losing_trades=0 if index == 0 else 1,
                gross_profit=0.0,
                gross_loss=0.0 if index == 0 else -1.0,
                max_consecutive_losses=0 if index == 0 else 1,
                final_balance=1000.0,
                guardrail_summary={
                    "rejected_fill_count": 0,
                    "zero_size_entry_count": 0,
                    "invalid_stop_distance_count": 0,
                    "max_required_leverage": 0.0,
                    "flags": 0,
                },
                sharpe_daily=float("nan"),
                sharpe_daily_observations=None if daily_none else 0,
                sharpe_daily_active_days=None if daily_none else 0,
                sqn=float("nan"),
                backend_kind="compiled_numba",
            )
        )
    return tuple(rows)


def fake_execute(plan: GridV2Plan, *_args: Any, **_kwargs: Any) -> Any:
    return SimpleNamespace(
        rows=fake_rows(plan),
        selected=(),
        metadata={
            "backend_kind": "compiled_numba",
            "compiled_batch_used": True,
            "compiled_execution_mode": "stacked",
            "compiled_config_packing": "table",
            "compiled_unavailable_reason": None,
        },
    )
