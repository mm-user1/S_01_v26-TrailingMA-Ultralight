from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.engine_v2.profile import active_parameter_names, inactive_parameter_names
from core.engine_v2.compiled_kernel import evaluate_compiled_batch
from core.engine_v2.runner import run_v2_strategy
from strategies.s06_r_trend_v02_b2.signals import (
    S06B2Params,
    build_s06_b2_execution_data,
)
from strategies.s06_r_trend_v06_4_a2_b2 import strategy
from strategies.s06_r_trend_v06_4_a2_b2.signals import S06V064A2Params


def _frame(length: int = 180) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=length, freq="30min", tz="UTC")
    base = 100.0 + np.sin(np.arange(length) / 5.0) * 8.0 + np.arange(length) * 0.01
    return pd.DataFrame(
        {
            "Open": base - 0.2,
            "High": base + 1.3,
            "Low": base - 1.1,
            "Close": base,
            "Volume": np.arange(length, dtype=float) + 1.0,
        },
        index=index,
    )


def _independent_pine_atr(frame: pd.DataFrame, length: int) -> np.ndarray:
    high = frame["High"].to_numpy(dtype=float)
    low = frame["Low"].to_numpy(dtype=float)
    close = frame["Close"].to_numpy(dtype=float)
    previous = np.empty_like(close)
    previous[0] = np.nan
    previous[1:] = close[:-1]
    true_range = np.maximum.reduce((high - low, np.abs(high - previous), np.abs(low - previous)))
    true_range[0] = high[0] - low[0]
    result = np.full(len(frame), np.nan)
    result[length - 1] = true_range[:length].mean()
    for index in range(length, len(frame)):
        result[index] = (result[index - 1] * (length - 1) + true_range[index]) / length
    return result


def test_identity_profile_mapping_and_active_only_bindings_are_exact():
    config = strategy.load_config()
    profile = strategy.load_profile()

    assert (config["id"], config["name"], config["version"], config["engine"]) == (
        "s06_r_trend_v06_4_a2_b2",
        "S06 R-Trend v06-4-A2 B2",
        "v06-4-a2-b2",
        "v2",
    )
    assert config["parameters"]["trailMode"]["default"] == "R Trail"
    assert config["parameters"]["trailMode"]["options"] == [
        "Off (Bracket)", "R Trail", "Chandelier Exit", "Fixed-AF SAR"
    ]
    assert config["parameters"]["trailMode"]["optimize"] == {"enabled": False}
    assert config["execution"]["variantSelector"] == {
        "param": "trailMode",
        "mapping": {
            "Off (Bracket)": "bracket",
            "R Trail": "r_trail",
            "Chandelier Exit": "chandelier",
            "Fixed-AF SAR": "fixed_af_sar",
        },
    }
    assert list(profile.variants) == ["bracket", "r_trail", "chandelier", "fixed_af_sar"]
    assert profile.validation_warnings == ()
    assert all("depends_on" not in spec for spec in config["parameters"].values())

    expected_active = {
        "Off (Bracket)": {"stopRR"},
        "R Trail": {"trailRR", "trailDistanceR"},
        "Chandelier Exit": {"trailRR", "chandelierATRLength", "chandelierATRMult"},
        "Fixed-AF SAR": {"trailRR", "sarSpeed"},
    }
    all_variant_fields = {
        "stopRR", "trailRR", "trailDistanceR", "chandelierATRLength",
        "chandelierATRMult", "sarSpeed",
    }
    for trail_mode, expected in expected_active.items():
        params = strategy.normalized_params({"trailMode": trail_mode})
        active = active_parameter_names(profile, params)
        inactive = inactive_parameter_names(profile, params)
        assert active & all_variant_fields == expected
        assert inactive & all_variant_fields == all_variant_fields - expected


def test_aliases_collapse_to_canonical_identity_and_inactive_values_are_inert():
    normalized = strategy.normalized_params({"fastSmoothing": 9, "slowSmoothing": 5})
    assert normalized["fastSmooth"] == 9
    assert normalized["slowSmooth"] == 5
    assert "fastSmoothing" not in normalized and "slowSmoothing" not in normalized

    params = S06V064A2Params.from_dict(
        {
            "trailMode": "R Trail",
            "stopRR": "inactive-bad",
            "chandelierATRLength": "inactive-bad",
            "chandelierATRMult": "inactive-bad",
            "sarSpeed": "inactive-bad",
        }
    )
    assert params.stopRR == 3.0
    assert params.chandelierATRLength == 14
    assert params.chandelierATRMult == 3.0
    assert params.sarSpeed == 0.01


@pytest.mark.parametrize(
    ("trail_mode", "inactive_fields"),
    [
        (
            "Off (Bracket)",
            ("trailRR", "trailDistanceR", "chandelierATRLength", "chandelierATRMult", "sarSpeed"),
        ),
        (
            "R Trail",
            ("stopRR", "chandelierATRLength", "chandelierATRMult", "sarSpeed"),
        ),
        (
            "Chandelier Exit",
            ("stopRR", "trailDistanceR", "sarSpeed"),
        ),
        (
            "Fixed-AF SAR",
            ("stopRR", "trailDistanceR", "chandelierATRLength", "chandelierATRMult"),
        ),
    ],
)
def test_direct_execution_canonicalizes_every_inactive_variant_field(
    trail_mode, inactive_fields
):
    frame = _frame()
    common = {"trailMode": trail_mode, "entryMode": "Trend @ Square", "dateFilter": False}
    expected = strategy.S06RTrendV064A2B2.run(frame, common)
    actual = strategy.S06RTrendV064A2B2.run(
        frame,
        {**common, **dict.fromkeys(inactive_fields, "inactive-bad")},
    )

    assert actual.trades == expected.trades
    np.testing.assert_array_equal(actual.equity_curve, expected.equity_curve)
    np.testing.assert_array_equal(actual.balance_curve, expected.balance_curve)


@pytest.mark.parametrize(
    ("trail_mode", "field"),
    [
        ("Off (Bracket)", "stopRR"),
        ("R Trail", "trailRR"),
        ("R Trail", "trailDistanceR"),
        ("Chandelier Exit", "chandelierATRLength"),
        ("Chandelier Exit", "chandelierATRMult"),
        ("Fixed-AF SAR", "sarSpeed"),
    ],
)
def test_direct_execution_rejects_malformed_active_fields(trail_mode, field):
    with pytest.raises(ValueError, match=field):
        strategy.S06RTrendV064A2B2.run(
            _frame(),
            {"trailMode": trail_mode, field: "active-bad", "dateFilter": False},
        )


@pytest.mark.parametrize("value", [2, 3, 4, 5, 2.0, 3.0, 5.0, "2", "2.0", "3", "5.0"])
def test_stop_lp_exact_integral_forms_pass_parser_and_direct_execution(value):
    assert S06V064A2Params.from_dict({"stopLP": value}).stopLP == int(float(value))
    strategy.S06RTrendV064A2B2.run(
        _frame(),
        {"trailMode": "Off (Bracket)", "stopLP": value, "dateFilter": False},
    )


@pytest.mark.parametrize(
    "value",
    [0, -1, 2.9, "2.9", True, False, float("nan"), float("inf"), float("-inf"), "bad"],
)
def test_stop_lp_nonpositive_fractional_boolean_nonfinite_and_nonnumeric_forms_are_rejected(
    value,
):
    with pytest.raises(ValueError, match="stopLP"):
        S06V064A2Params.from_dict({"stopLP": value})
    with pytest.raises(ValueError, match="stopLP"):
        strategy.S06RTrendV064A2B2.run(
            _frame(),
            {"trailMode": "Off (Bracket)", "stopLP": value, "dateFilter": False},
        )


@pytest.mark.parametrize(
    ("trail_mode", "field", "value"),
    [
        ("Off (Bracket)", "stopRR", 1.0),
        ("R Trail", "trailRR", 0.5),
        ("R Trail", "trailDistanceR", 3.5),
        ("Chandelier Exit", "chandelierATRLength", 21),
        ("Chandelier Exit", "chandelierATRMult", 7.0),
        ("Fixed-AF SAR", "sarSpeed", 0.025),
    ],
)
def test_active_variant_legal_bounds_are_enforced(trail_mode, field, value):
    with pytest.raises(ValueError, match=field):
        S06V064A2Params.from_dict({"trailMode": trail_mode, field: value})


@pytest.mark.parametrize("entry_mode", ["Reversal @ Triangle", "Trend @ Square"])
def test_shared_signals_and_initial_stop_arrays_match_existing_s06_exactly(entry_mode):
    frame = _frame()
    common = {
        "entryMode": entry_mode,
        "fastLength": 21,
        "fastSmooth": 7,
        "slowLength": 112,
        "slowSmooth": 3,
        "thresholdOS": 20,
        "thresholdOB": 20,
        "stopLP": 4,
    }
    old = build_s06_b2_execution_data(frame, S06B2Params.from_dict(common))
    new = strategy.build_v2_execution_data(
        frame,
        strategy.normalized_params({**common, "trailMode": "Off (Bracket)", "dateFilter": False}),
    )

    np.testing.assert_array_equal(new.signals.long_entries, old.signals.long_entries)
    np.testing.assert_array_equal(new.signals.short_entries, old.signals.short_entries)
    for name in ("atr", "rolling_low", "rolling_high"):
        np.testing.assert_array_equal(getattr(new, name), getattr(old, name))
    assert new.chandelier_atr is None
    assert np.isnan(new.trail_long).all() and np.isnan(new.trail_short).all()


@pytest.mark.parametrize("length", [14, 28])
def test_chandelier_atr_matches_independent_pine_rma_oracle(length):
    frame = _frame(80)
    data = strategy.build_v2_execution_data(
        frame,
        strategy.normalized_params(
            {"trailMode": "Chandelier Exit", "chandelierATRLength": length, "dateFilter": False}
        ),
    )
    assert data.chandelier_atr is not None
    np.testing.assert_allclose(
        data.chandelier_atr,
        _independent_pine_atr(frame, length),
        rtol=1e-15,
        atol=1e-15,
        equal_nan=True,
    )


def test_non_chandelier_preparation_never_computes_ma_or_chandelier(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("MA/Chandelier computation must remain request-gated")

    monkeypatch.setattr("strategies.s06_r_trend_v02_b2.signals.trail_ma", forbidden)
    monkeypatch.setattr("strategies.s06_r_trend_v06_4_a2_b2.signals.pine_atr", forbidden)
    data = strategy.build_v2_execution_data(
        _frame(),
        strategy.normalized_params({"trailMode": "Fixed-AF SAR", "dateFilter": False}),
    )
    assert data.chandelier_atr is None
    assert strategy.DATAPREP_CACHE_PARAM_NAMES == (
        *strategy.SIGNAL_CACHE_PARAM_NAMES,
        "stopLP",
        "chandelierATRLength",
    )
    assert not {"trailMAType", "trailMALength", "trailMAOffsetEx"} & set(
        strategy.DATAPREP_CACHE_PARAM_NAMES
    )


def test_config_copy_is_caller_owned():
    first = strategy.load_config()
    second = strategy.load_config()
    first["parameters"]["trailMode"]["default"] = "Off (Bracket)"
    assert second["parameters"]["trailMode"]["default"] == "R Trail"


@pytest.mark.parametrize(
    "trail_mode",
    ["Off (Bracket)", "R Trail", "Chandelier Exit", "Fixed-AF SAR"],
)
def test_each_exit_variant_runs_through_direct_and_compiled_backtest(trail_mode):
    params = strategy.normalized_params(
        {"trailMode": trail_mode, "entryMode": "Trend @ Square", "dateFilter": False}
    )
    data = strategy.build_v2_execution_data(_frame(), params)
    profile = strategy.load_profile()
    direct = run_v2_strategy(
        data=data, profile=profile, params=params, trade_start_idx=0,
        compute_max_drawdown_mtm=True,
    )
    compiled = evaluate_compiled_batch(
        data=data, profile=profile, params_batch=[params], trade_start_idx=0,
        compute_max_drawdown_mtm=True,
    )
    assert compiled.outputs.shape == (1, 26)
    assert compiled.max_drawdown_mtm_pct is not None
    assert compiled.outputs[0, 2] == direct.strategy_result.total_trades
    assert compiled.outputs[0, 0] == pytest.approx(direct.strategy_result.net_profit_pct)
    assert compiled.max_drawdown_mtm_pct[0] == pytest.approx(direct.max_drawdown_mtm_pct)
