import json
from pathlib import Path

import pandas as pd
from dataclasses import asdict

import pytest


from core.backtest_engine import load_data, prepare_dataset_with_warmup
from strategies.s01_trailing_ma.strategy import S01Params, S01TrailingMA

DATA_PATH = str(Path("data") / "raw" / "OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv")


@pytest.fixture(scope="module")
def baseline_metrics():
    baseline_path = Path("data") / "baseline" / "s01_metrics.json"
    with open(baseline_path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def test_data():
    return load_data(DATA_PATH)


@pytest.fixture(scope="module")
def baseline_params(baseline_metrics):
    return baseline_metrics["parameters"]


@pytest.fixture(scope="module")
def baseline_warmup(baseline_metrics):
    return baseline_metrics.get("warmup_bars", 1000)


class TestS01Strategy:
    def test_params_dataclass_from_dict(self, baseline_params):
        params = S01Params.from_dict(baseline_params)
        assert params.maType == baseline_params["maType"]
        assert params.maLength == baseline_params["maLength"]
        assert params.trailMaType == baseline_params["trailMaType"]
        assert params.closeCountLong == baseline_params["closeCountLong"]
        assert params.closeCountShort == baseline_params["closeCountShort"]

    def test_params_dataclass_to_dict(self, baseline_params):
        params = S01Params.from_dict(baseline_params)
        params_dict = asdict(params)
        assert params_dict["maType"] == baseline_params["maType"]
        assert params_dict["maLength"] == baseline_params["maLength"]
        assert params_dict["trailMaType"] == baseline_params["trailMaType"]
        assert params_dict["closeCountLong"] == baseline_params["closeCountLong"]
        assert params_dict["closeCountShort"] == baseline_params["closeCountShort"]

    def test_strategy_runs_without_error(
        self, test_data, baseline_params, baseline_warmup, baseline_metrics
    ):
        start_ts = pd.Timestamp(baseline_params["start"], tz="UTC")
        end_ts = pd.Timestamp(baseline_params["end"], tz="UTC")
        df_prepared, trade_start_idx = prepare_dataset_with_warmup(
            test_data, start_ts, end_ts, warmup_bars=baseline_warmup
        )

        result = S01TrailingMA.run(df_prepared, baseline_params, trade_start_idx)
        assert result is not None
        assert len(result.equity_curve) > 0
        assert len(result.balance_curve) > 0
        assert result.total_trades == baseline_metrics["total_trades"]

    def test_strategy_matches_baseline(
        self, test_data, baseline_params, baseline_metrics, baseline_warmup
    ):
        start_ts = pd.Timestamp(baseline_params["start"], tz="UTC")
        end_ts = pd.Timestamp(baseline_params["end"], tz="UTC")
        df_prepared, trade_start_idx = prepare_dataset_with_warmup(
            test_data, start_ts, end_ts, warmup_bars=baseline_warmup
        )

        result = S01TrailingMA.run(df_prepared, baseline_params, trade_start_idx)

        assert result.net_profit_pct == pytest.approx(
            baseline_metrics["net_profit_pct"], rel=0, abs=1e-6
        )
        assert result.max_drawdown_pct == pytest.approx(
            baseline_metrics["max_drawdown_pct"], rel=0, abs=1e-6
        )
        assert result.total_trades == baseline_metrics["total_trades"]


class TestS01MATypes:
    @pytest.mark.parametrize(
        "ma_type",
        ["SMA", "EMA", "HMA", "WMA", "ALMA", "KAMA", "TMA", "T3", "DEMA", "VWMA", "VWAP"],
    )
    def test_ma_type_compatibility(
        self, test_data, baseline_params, baseline_warmup, ma_type
    ):
        params = baseline_params.copy()
        params["maType"] = ma_type
        params["trailMaType"] = ma_type

        start_ts = pd.Timestamp(params["start"], tz="UTC")
        end_ts = pd.Timestamp(params["end"], tz="UTC")
        df_prepared, trade_start_idx = prepare_dataset_with_warmup(
            test_data, start_ts, end_ts, warmup_bars=baseline_warmup
        )

        first_run = S01TrailingMA.run(df_prepared, params, trade_start_idx)
        second_run = S01TrailingMA.run(df_prepared, params, trade_start_idx)

        assert abs(first_run.net_profit_pct - second_run.net_profit_pct) < 1e-9
        assert first_run.total_trades == second_run.total_trades
