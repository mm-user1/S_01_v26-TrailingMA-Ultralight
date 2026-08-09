from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.backtest_engine import load_data, prepare_dataset_with_warmup  # noqa: E402
from core.grid_engine import rank_grid_results  # noqa: E402
from core.optuna_engine import OptimizationConfig  # noqa: E402
from strategies.s03_reversal_v10 import fast_grid as s03_v10  # noqa: E402
from strategies.s03_reversal_v11 import fast_grid as s03_v11  # noqa: E402
from strategies.s06_r_trend_v02 import fast_grid as s06  # noqa: E402


ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "raw" / "OKX_SUIUSDT.P, 30 2025.01.01-2026.02.01.csv"
TOLERANCES = {
    "net_profit_pct_abs": 0.001,
    "max_drawdown_pct_abs": 0.001,
    "romad_abs": 0.005,
    "win_rate_abs": 0.001,
    "total_trades_abs": 0.0,
    "winning_trades_abs": 0.0,
    "losing_trades_abs": 0.0,
    "max_consecutive_losses_abs": 0.0,
    "sharpe_ratio_abs": 1e-12,
    "sharpe_ratio_rel": 1e-9,
    "sqn_abs": 1e-12,
    "sqn_rel": 1e-9,
}


@pytest.fixture(scope="module")
def prepared_cases():
    if not DATA_PATH.exists():
        pytest.skip(f"Reference data not found: {DATA_PATH}")
    raw = load_data(str(DATA_PATH))
    start = pd.Timestamp("2025-02-01T00:00:00Z")
    end = pd.Timestamp("2026-02-01T00:00:00Z")
    frame, trade_start_idx = prepare_dataset_with_warmup(raw, start, end, 1000)
    base_params = {
        "dateFilter": True,
        "start": start,
        "end": end,
        "maType3": "SMA",
        "maLength3": 75,
        "maOffset3": 0.2,
        "useCloseCount": True,
        "closeCountLong": 7,
        "closeCountShort": 5,
        "useTBands": True,
        "tBandLongPct": 1.0,
        "tBandShortPct": 1.3,
        "contractSize": 0.01,
        "initialCapital": 100.0,
        "commissionPct": 0.05,
    }

    cases: dict[str, tuple[Any, pd.DataFrame, int, Any, Any]] = {}
    for name, backend, params in (
        ("s03_v10", s03_v10, dict(base_params)),
        (
            "s03_v11",
            s03_v11,
            {
                **base_params,
                "useEmergencySL": True,
                "emergencySlPct": 10.0,
                "emergencySlUpdateBars": 16,
            },
        ),
    ):
        candidate = backend.GridCandidate(
            candidate_id=1,
            mode="both",
            params=params,
            semantic_key=backend.candidate_semantic_key("both", params),
            generation_mode="test",
            diversity_group="both|SMA|75",
        )
        data = backend.prepare_fast_data(frame, trade_start_idx, [candidate])
        cases[name] = (backend, frame, trade_start_idx, data, [candidate])

    s06_start = pd.Timestamp("2025-08-01T00:00:00Z")
    s06_end = pd.Timestamp("2025-12-01T00:00:00Z")
    s06_frame, s06_trade_start_idx = prepare_dataset_with_warmup(
        raw,
        s06_start,
        s06_end,
        1000,
    )
    s06_fixed = {
        "dateFilter": True,
        "start": s06_start.isoformat(),
        "end": s06_end.isoformat(),
        "entryMode": "Reversal @ Triangle",
        "enableLong": True,
        "enableShort": True,
        "fastLength": 21,
        "fastSmoothing": 7,
        "slowLength": 112,
        "slowSmoothing": 3,
        "thresholdOS": 20,
        "thresholdOB": 20,
        "stopX": 2.0,
        "stopRR": 3.0,
        "stopLP": 2,
        "stopMaxPct": 6.0,
        "stopMaxDays": 6,
        "riskPerTrade": 2.0,
        "contractSize": 0.01,
        "useTrailMA": False,
        "trailRR": 1.0,
        "trailMAType": "SMA",
        "trailMALength": 150,
        "trailMAOffsetEx": 0.0,
        "initialCapital": 100.0,
        "commissionPct": 0.05,
    }
    grid_params = (
        "stopX",
        "stopRR",
        "stopLP",
        "stopMaxPct",
        "stopMaxDays",
        "trailRR",
        "trailMAType",
        "trailMALength",
        "trailMAOffsetEx",
    )
    config = OptimizationConfig(
        csv_file=str(DATA_PATH),
        strategy_id="s06_r_trend_v02",
        enabled_params={name: False for name in (*grid_params, "thresholdOS", "thresholdOB")},
        param_ranges={},
        param_types={},
        fixed_params=s06_fixed,
        worker_processes=1,
        warmup_bars=1000,
        optimization_mode="grid",
        objectives=["net_profit_pct"],
        grid_fast_objectives=["net_profit_pct"],
        grid_enabled_modes=["bracket"],
        grid_budget=1,
        grid_seed=42,
        grid_top_candidates=1,
    )
    space = s06.build_parameter_space(config)
    allocation = s06.build_allocation(config, space, None)
    candidate_set = s06.generate_candidates(config, space, allocation, seed=42)
    s06_data = s06.prepare_fast_data(
        s06_frame,
        s06_trade_start_idx,
        candidate_set.candidates,
    )
    cases["s06"] = (
        s06,
        s06_frame,
        s06_trade_start_idx,
        s06_data,
        candidate_set.candidates,
    )
    return cases


@pytest.mark.parametrize("case_name", ["s03_v10", "s03_v11", "s06"])
def test_v1_fast_sharpe_and_sqn_are_conditional_and_match_reference(prepared_cases, case_name):
    backend, frame, trade_start_idx, data, candidates = prepared_cases[case_name]
    candidate = candidates[0]

    disabled = backend.evaluate_candidates(data, [candidate])[0]
    sharpe_only = backend.evaluate_candidates(data, [candidate], compute_sharpe=True)[0]
    sqn_only = backend.evaluate_candidates(data, [candidate], compute_sqn=True)[0]
    both = backend.evaluate_candidates(
        data,
        [candidate],
        compute_sharpe=True,
        compute_sqn=True,
    )[0]

    assert disabled.sharpe_ratio is None
    assert disabled.sqn is None
    assert sharpe_only.sharpe_ratio is not None and math.isfinite(sharpe_only.sharpe_ratio)
    assert sharpe_only.sqn is None
    assert sqn_only.sharpe_ratio is None
    assert sqn_only.sqn is not None and math.isfinite(sqn_only.sqn)
    assert both.sharpe_ratio == sharpe_only.sharpe_ratio
    assert both.sqn == sqn_only.sqn
    projected = backend._result_metric_dict(both)
    assert projected["sharpe_ratio"] == both.sharpe_ratio
    assert projected["sqn"] == both.sqn
    if case_name == "s06":
        assert not hasattr(disabled, "fast_metrics")
        assert not hasattr(both, "fast_metrics")
    else:
        assert both.fast_metrics["sharpe_ratio"] == both.sharpe_ratio
        assert both.fast_metrics["sqn"] == both.sqn
    assert not hasattr(both, "dsr_track_length") or both.dsr_track_length is None

    validated = backend.validate_selected_candidates(
        frame,
        trade_start_idx,
        [both],
        tolerances=TOLERANCES,
        fail_on_error=True,
    )[0]
    assert validated.validation_status == "passed"
    assert validated.validation_diffs["sharpe_ratio"]["passed"] is True
    assert validated.validation_diffs["sqn"]["passed"] is True
    assert validated.fast_metrics["sharpe_ratio"] == both.sharpe_ratio
    assert validated.fast_metrics["sqn"] == both.sqn
    assert both.sharpe_ratio == pytest.approx(validated.sharpe_ratio, rel=1e-9, abs=1e-12)
    assert both.sqn == pytest.approx(validated.sqn, rel=1e-9, abs=1e-12)

    for primary in ("sharpe_ratio", "sqn"):
        ranked = rank_grid_results(
            [both],
            objectives=[primary],
            primary_objective=None,
            constraints=[],
            stage_label=f"internal {case_name} {primary}",
        )
        assert ranked == [both]
        assert ranked[0].objective_values == [getattr(both, primary)]


@pytest.mark.parametrize("case_name", ["s03_v10", "s03_v11", "s06"])
def test_v1_dsr_reuses_sharpe_without_enabling_sqn(prepared_cases, case_name):
    backend, _frame, _trade_start_idx, data, candidates = prepared_cases[case_name]
    candidate = candidates[0]

    result = backend.evaluate_candidates(data, [candidate], needs_dsr=True)[0]

    assert result.sharpe_ratio is not None
    assert result.sqn is None
    assert result.dsr_track_length >= 3
    assert result.dsr_skewness is not None
    assert result.dsr_kurtosis is not None
    corrected_calendar_pins = {
        "s03_v10": (0.5098869911832709, 13, 0.9757645959064098, 3.026467558616449),
        "s03_v11": (0.5482407780597649, 13, 0.850371966940852, 2.9545541261841404),
        "s06": (-0.6106162762799014, 5, 0.35980610920011097, 1.5073494122398496),
    }
    sharpe, track_length, skewness, kurtosis = corrected_calendar_pins[case_name]
    assert result.sharpe_ratio == pytest.approx(sharpe, rel=1e-12, abs=1e-12)
    assert result.dsr_track_length == track_length
    assert result.dsr_skewness == pytest.approx(skewness, rel=1e-10, abs=1e-12)
    assert result.dsr_kurtosis == pytest.approx(kurtosis, rel=1e-10, abs=1e-12)


@pytest.mark.parametrize(
    ("start", "end", "expected_track_length", "expect_sharpe"),
    [
        ("2025-09-01T00:00:00Z", "2025-09-30T23:59:59Z", 1, False),
        ("2025-08-01T00:00:00Z", "2025-09-30T23:59:59Z", 2, True),
    ],
)
def test_v1_fast_short_real_windows_use_only_real_calendar_months(
    prepared_cases,
    start,
    end,
    expected_track_length,
    expect_sharpe,
):
    backend, _frame, _trade_start_idx, _data, candidates = prepared_cases["s03_v10"]
    raw = load_data(str(DATA_PATH))
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    frame, trade_start_idx = prepare_dataset_with_warmup(raw, start_ts, end_ts, 1000)
    params = {**candidates[0].params, "start": start_ts, "end": end_ts}
    candidate = backend.GridCandidate(
        candidate_id=1,
        mode="both",
        params=params,
        semantic_key=backend.candidate_semantic_key("both", params),
        generation_mode="test",
        diversity_group="both|SMA|75",
    )
    data = backend.prepare_fast_data(frame, trade_start_idx, [candidate])

    result = backend.evaluate_candidates(data, [candidate], needs_dsr=True)[0]

    assert result.dsr_track_length == expected_track_length
    assert (result.sharpe_ratio is not None) is expect_sharpe
    assert result.dsr_skewness is None
    assert result.dsr_kurtosis is None


def test_all_v1_close_sites_update_sqn_once():
    expected_sites = {
        ROOT / "src" / "strategies" / "s03_reversal_v10" / "fast_grid.py": 3,
        ROOT / "src" / "strategies" / "s03_reversal_v11" / "fast_grid.py": 4,
        ROOT / "src" / "strategies" / "s06_r_trend_v02" / "fast_grid.py": 3,
    }
    for path, expected in expected_sites.items():
        source = path.read_text(encoding="utf-8")
        assert source.count("total_trades += 1") == expected
        assert source.count("sqn_count += 1") == expected


@pytest.mark.parametrize("backend", [s03_v10, s03_v11, s06])
def test_v1_fast_metric_comparator_has_strict_three_state_semantics(backend):
    comparator = backend._optional_metric_matches

    assert comparator(None, None, absolute_tolerance=1e-12, relative_tolerance=1e-9) == (
        True,
        0.0,
    )
    assert comparator(None, 0.0, absolute_tolerance=1e-12, relative_tolerance=1e-9)[0] is False
    assert comparator(0.0, None, absolute_tolerance=1e-12, relative_tolerance=1e-9)[0] is False
    assert comparator(
        1_000_000_000.0,
        1_000_000_000.5,
        absolute_tolerance=1e-12,
        relative_tolerance=1e-9,
    )[0] is True
    assert comparator(
        1.0,
        1.01,
        absolute_tolerance=1e-12,
        relative_tolerance=1e-9,
    )[0] is False
