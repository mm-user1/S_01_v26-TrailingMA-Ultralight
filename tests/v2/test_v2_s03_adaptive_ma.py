"""Production strategy, external residual and reduced Grid certification."""

import csv
import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.backtest_engine import load_data, prepare_dataset_with_warmup
from core.engine_v2.runner import run_v2_strategy
from core.grid_v2 import (
    COMPILED_BATCH_KIND, GridV2Settings, GridV2StrategyHooks, GridV2PlanReuseCache,
    build_grid_v2_plan, preview_grid_v2_counts, execute_grid_v2_candidates,
)
from strategies import get_strategy
from strategies.s03_reversal_v16_4_a_adaptive_ma_b2 import strategy as strategy
from strategies.s03_reversal_v16_4_a_adaptive_ma_b2 import signals
from indicators.ma import kama

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "data/baseline_v2/s03_reversal_v16_4_a_adaptive_ma"
REFERENCES = ("reference_a_kama", "reference_b_supersmoother", "reference_c_frama+sl10", "reference_d_dsma+sl10")
MA_TYPES = ("KAMA", "SuperSmoother", "FRAMA", "DSMA")
TIES = ("symmetricLongShort",)
NET = (36.65663378600007, 37.35478997499999, -2.594669402000008, 8.600414829999934)
COUNTS = (63, 66, 62, 53)
DD = (21.320000857549502, 19.481348177057505, 27.289743307351365, 23.064183519914025)


def frame(n=400):
    x = np.arange(n)
    close = 100 + 15*np.sin(x/17) + 4*np.sin(x/3) + x/100
    return pd.DataFrame({"Open": close, "High": close+2, "Low": close-2, "Close": close, "Volume": 100.0},
                        index=pd.date_range("2025-01-01", periods=n, freq="30min", tz="UTC"))


@pytest.fixture(scope="module")
def prepared():
    manifest = json.loads((BASELINE / "dataset.json").read_text())
    market = load_data(ROOT / manifest["market_data"]["path"])
    assert len(market) == 19056
    assert (market.index.to_series().diff().dropna() == pd.Timedelta(minutes=30)).all()
    return prepare_dataset_with_warmup(market, pd.Timestamp("2025-08-01T00:00Z"), pd.Timestamp("2025-12-01T00:00Z"), 1000)


def reference_params(ref):
    raw = json.loads((BASELINE / ref / "params.json").read_text())["strategy_inputs"]
    return strategy.normalized_params({**raw, "start": "2025-08-01T00:00Z", "end": "2025-12-01T00:00Z"})


def test_frozen_hashes_and_discovery():
    manifest = json.loads((BASELINE / "dataset.json").read_text())
    for asset in [*manifest["asset_hashes"], manifest["pine_source"], manifest["market_data"]]:
        assert hashlib.sha256((ROOT / asset["path"]).read_bytes()).hexdigest() == asset["sha256"]
    assert get_strategy(strategy.S03ReversalV164AAdaptiveMAB2.STRATEGY_ID) is strategy.S03ReversalV164AAdaptiveMAB2
    config = strategy.load_config()
    assert config["parameters"]["maType3"]["options"] == list(MA_TYPES)
    assert tuple(n for n,s in config["parameters"].items() if s["role"] == "signal") == strategy.SIGNAL_CACHE_PARAM_NAMES
    assert strategy.DATAPREP_CACHE_PARAM_NAMES == strategy.SIGNAL_CACHE_PARAM_NAMES
    assert len(strategy.SIGNAL_CACHE_PARAM_NAMES) == 6
    assert strategy.load_profile().variant_selector.user_facing is False


@pytest.mark.parametrize("name,value", [("maLength3", 24), ("maLength3", 251), ("maLength3", 25.5), ("closeCountLong", 0), ("closeCountShort", 8), ("tBandLongPct", float("nan")), ("commissionPct", -1), ("initialCapital", float("inf")), ("maType3", "SUPERSMOOTHER")])
def test_invalid_parameters(name, value):
    with pytest.raises(ValueError):
        strategy.normalized_params({name: value})


@pytest.mark.parametrize("length", [25, 26, 125, 250])
@pytest.mark.parametrize("start", [0, 37, 378])
def test_shared_kama_exact_reuse(monkeypatch, length, start):
    df = frame().iloc[start:]
    calls = []
    def checked(series, n):
        calls.append(n)
        return kama(series, n)
    monkeypatch.setattr(signals.shared_ma, "kama", checked)
    actual = signals.moving_average(df, "KAMA", length)
    np.testing.assert_array_equal(actual, kama(df.Close, length).to_numpy())
    assert calls == [length]
    np.testing.assert_array_equal(actual[:min(length, len(df))], np.full(min(length, len(df)), df.Close.iloc[0]))


@pytest.mark.parametrize("ma_type", MA_TYPES)
@pytest.mark.parametrize("n", [0, 1, 27, 300])
def test_prefix_integer_batch_and_ownership(ma_type, n):
    df = frame().round().astype(int)
    before = df.copy(deep=True)
    params = strategy.normalized_params({"maType3": ma_type, "maLength3": 25})
    full = strategy.build_v2_execution_data(df, params)
    part = strategy.build_v2_execution_data(df.iloc[:n], params)
    np.testing.assert_array_equal(full.signals.long_entries[:n], part.signals.long_entries)
    assert part.close.dtype == np.float64
    batch = strategy.build_v2_execution_data_batch(df.iloc[:n], [params, {**params, "commissionPct": 0.1}])
    for data in batch:
        np.testing.assert_array_equal(data.signals.short_entries, part.signals.short_entries)
    pd.testing.assert_frame_equal(before, df)


def test_band_memory_precedence_and_counter_equality():
    close = np.array([100., 102., 100.5, 100., 98., 100.])
    high = np.array([100., 103., 101., 102., 99., 102.])
    low = np.array([100., 101., 100., 98., 97., 98.])
    average = np.full(6, 100.)
    np.testing.assert_array_equal(signals.band_states(close, high, low, average, 1., 1.), [0,1,1,-1,-1,-1])
    long, short = signals.close_counters(close, average)
    np.testing.assert_array_equal(long, [0,1,2,0,0,0])
    np.testing.assert_array_equal(short, [0,0,0,0,1,0])


@pytest.mark.parametrize("ma_type", MA_TYPES[1:])
def test_independent_filter_formula(ma_type):
    df = frame(95)
    length = 25
    close = df.Close.to_numpy()
    expected = close.copy()
    filt = np.zeros(len(close))
    n = length + length % 2
    for i in range(len(close)):
        if ma_type == "FRAMA":
            if i < n: continue
            a = df.iloc[i-n//2+1:i+1]
            b = df.iloc[i-n+1:i-n//2+1]
            all_bars = df.iloc[i-n+1:i+1]
            ranges = ((a.High.max()-a.Low.min())/(n//2), (b.High.max()-b.Low.min())/(n//2), (all_bars.High.max()-all_bars.Low.min())/n)
            dimension = math.log2((ranges[0]+ranges[1])/ranges[2]) if min(ranges)>0 else 1.
            alpha = min(1., max(.01, math.exp(-4.6*(dimension-1))))
            expected[i] = alpha*close[i]+(1-alpha)*expected[i-1]
        else:
            period = length if ma_type == "SuperSmoother" else length/2
            a = math.exp(-math.sqrt(2)*math.pi/period)
            c2,c3 = 2*a*math.cos(math.sqrt(2)*math.pi/period), -a*a
            c1=1-c2-c3
            if ma_type == "SuperSmoother" and i >= 3:
                expected[i] = c1*(close[i]+close[i-1])/2+c2*expected[i-1]+c3*expected[i-2]
            if ma_type == "DSMA":
                if i >= 2:
                    zero = close[i]-close[i-2]
                    prev = close[i-1]-close[i-3] if i >= 3 else 0
                    filt[i]=c1*(zero+prev)/2+c2*filt[i-1]+c3*filt[i-2]
                if i >= length+4:
                    rms = np.sqrt(np.mean(filt[i-length+1:i+1]**2))
                    alpha = abs(filt[i]/rms)*5/length if rms else 0
                    expected[i]=alpha*close[i]+(1-alpha)*expected[i-1]
    np.testing.assert_allclose(signals.moving_average(df, ma_type, length), expected, rtol=1e-13, atol=1e-12)
    constant = df.copy(); constant.loc[:, ["Open", "High", "Low", "Close"]] = 100.
    np.testing.assert_allclose(signals.moving_average(constant, ma_type, length), 100., atol=1e-10)


@pytest.mark.parametrize("ref", REFERENCES)
def test_external_reference_residuals(ref, prepared):
    df, start = prepared
    assert (len(df), start) == (6857, 1000)
    params = reference_params(ref)
    result = run_v2_strategy(data=strategy.build_v2_execution_data(df, params), profile=strategy.load_profile(), params=params, trade_start_idx=start)
    index = REFERENCES.index(ref)
    assert result.basic_metrics.total_trades == COUNTS[index]
    assert result.basic_metrics.net_profit_pct == pytest.approx(NET[index], abs=1e-10)
    assert result.basic_metrics.max_drawdown_pct == pytest.approx(DD[index], abs=1e-10)
    frozen = json.loads((BASELINE / "merlin_expectations.json").read_text())["references"][ref]
    for family, metrics in (("basic",result.basic_metrics),("advanced",result.advanced_metrics)):
        for name, expected in frozen[family].items():
            actual = getattr(metrics,name)
            assert actual is None if expected is None else actual == pytest.approx(expected,rel=1e-9,abs=1e-10)
    np.testing.assert_allclose([t.size for t in result.strategy_result.trades],frozen["trade_sizes"],rtol=0,atol=1e-10)
    np.testing.assert_allclose([t.net_pnl for t in result.strategy_result.trades],frozen["trade_net_pnl"],rtol=1e-10,atol=1e-10)
    tv = list(csv.DictReader((BASELINE/ref/"trades_normalized_utc.csv").open(encoding="utf-8-sig")))
    stops = [t for t in result.strategy_result.trades if t.exit_reason == "Emergency SL"]
    assert len(stops) == [0,0,3,4][index]
    assert sum(t.direction == "long" for t in stops) == [0,0,1,1][index]
    residuals = {2: {8: 3.72573, 11: 3.68973, 47: 2.12113}, 3: {9: 3.69743, 31: 2.72217, 45: 2.09286}}.get(index, {})
    for number, (actual, expected) in enumerate(zip(result.strategy_result.trades, tv), 1):
        assert actual.direction == expected["direction"]
        assert actual.entry_time == pd.Timestamp(expected["entry_time_utc"])
        assert actual.entry_price == float(expected["entry_price_usdt"])
        if number == len(tv):
            assert actual.exit_time == pd.Timestamp("2025-12-01T00:00Z")
            assert actual.exit_price == pytest.approx(1.42443 if index == 3 else 1.423, abs=1e-12)
        else:
            assert actual.exit_time == pd.Timestamp(expected["exit_time_utc"])
            assert actual.exit_price == pytest.approx(residuals.get(number, float(expected["exit_price_usdt"])), abs=1e-12)
        if actual.exit_reason == "Emergency SL":
            fill = df.index.get_loc(actual.entry_time)
            exit_index = df.index.get_loc(actual.exit_time)
            factor = .9 if actual.direction == "long" else 1.1
            stop = actual.entry_price * factor
            for bar in range(fill + 16, exit_index, 16):
                candidate = float(df.Close.iloc[bar]) * factor
                stop = max(stop,candidate) if actual.direction == "long" else min(stop,candidate)
            opening = float(df.Open.iloc[exit_index])
            assert actual.exit_price == pytest.approx(min(opening,stop) if actual.direction == "long" else max(opening,stop),abs=1e-12)
        # Exported quantities have two decimals; one contract step is characterized separately.
        assert abs(actual.size-float(expected["size_qty"])) <= .01000000001


@pytest.mark.parametrize("ref", REFERENCES)
@pytest.mark.parametrize("emergency", [False,True])
def test_real_single_candidate_compiled_and_replay(ref,emergency,prepared):
    df,start=prepared
    params={**reference_params(ref),"useEmergencySL":emergency}
    plan=build_grid_v2_plan(strategy.load_config(),GridV2Settings(enabled_axes=(),top_n=1),params)
    run=execute_grid_v2_candidates(plan,df,start,GridV2StrategyHooks.from_strategy(strategy),compute_sharpe=True,compute_sqn=True,compute_sharpe_daily=True)
    reference=run_v2_strategy(data=strategy.build_v2_execution_data(df,params),profile=strategy.load_profile(),params=params,trade_start_idx=start,compute_sharpe_daily=True)
    assert run.metadata["compiled_batch_used"] and run.rows[0].status == "ok"
    assert run.rows[0].total_trades == reference.basic_metrics.total_trades
    assert run.rows[0].net_profit_pct == pytest.approx(reference.basic_metrics.net_profit_pct,abs=1e-10)
    assert run.rows[0].sharpe_daily == pytest.approx(reference.advanced_metrics.sharpe_daily,abs=1e-10)
    assert run.selected[0].row.candidate_id == 1


@pytest.mark.parametrize("ties,count", [((),196000),(TIES,2800)])
def test_preview_counts_without_population(monkeypatch, ties, count):
    import core.grid_v2 as grid
    def forbidden(*args, **kwargs): raise AssertionError("Preview materialized a population")
    monkeypatch.setattr(grid, "_build_candidate_table", forbidden)
    settings = GridV2Settings(enabled_tie_groups=ties)
    assert preview_grid_v2_counts(strategy.load_config(), settings).raw_candidate_count == count
    assert preview_grid_v2_counts(strategy.load_config(), settings, {"maType3_options":["DSMA"]}).raw_candidate_count == count//4
    axes = (*strategy.SIGNAL_CACHE_PARAM_NAMES, "emergencySlPct")
    assert preview_grid_v2_counts(strategy.load_config(), replace(settings, enabled_axes=axes), {"useEmergencySL":True}).raw_candidate_count == count*2


def small_config():
    config = strategy.load_config()
    for name in strategy.SIGNAL_CACHE_PARAM_NAMES[1:]:
        spec = config["parameters"][name]["optimize"]
        spec["max"] = spec["min"] + spec["step"]
    return config


def test_reversed_pair_preserves_filtered_canonical_order():
    config = small_config()
    config["optimization_rules"]["parameter_tie_groups"][0]["pairs"][0].reverse()
    names = list(config["parameters"])
    names.remove("tBandLongPct")
    names.insert(names.index("closeCountShort"), "tBandLongPct")
    config["parameters"] = {name: config["parameters"][name] for name in names}
    plain = build_grid_v2_plan(config)
    tied = build_grid_v2_plan(config, GridV2Settings(enabled_tie_groups=TIES))
    expected = [c.semantic_key for c in plain.candidates
                if c.params["closeCountLong"] == c.params["closeCountShort"]
                and c.params["tBandLongPct"] == c.params["tBandShortPct"]]
    assert [c.semantic_key for c in tied.candidates] == expected


def test_standard_optimize_study_roundtrip(tmp_path):
    from core import storage
    from core.grid_engine import run_grid_optimization
    from ui import server_services
    df = frame(600)
    csv_path = tmp_path / "s03.csv"
    df.insert(0, "time", df.index.as_unit("s").asi8)
    df.to_csv(csv_path, index=False)
    payload = {"strategy_id": strategy.S03ReversalV164AAdaptiveMAB2.STRATEGY_ID,
               "optimization_mode": "grid", "enabled_params": {"closeCountLong": True},
               "param_ranges": {"closeCountLong": [1, 2, 1]}, "param_types": {"closeCountLong": "int"},
               "fixed_params": {"dateFilter": False, "maType3": "SuperSmoother", "maLength3": 25},
               "grid_v2_enabled_tie_groups": list(TIES), "worker_processes": 1,
               "grid_top_candidates": 2, "grid_diversity_enabled": False,
               "objectives": ["net_profit_pct"], "grid_fast_objectives": ["net_profit_pct"]}
    config = server_services._build_optimization_config(str(csv_path), payload, 1, payload["strategy_id"])
    results, study_id = run_grid_optimization(config, save_study=True)
    assert results and study_id
    loaded = storage.load_study_from_db(study_id)
    saved = loaded["study"]["config_json"]
    assert saved["grid_v2_enabled_tie_groups"] == saved["grid_config"]["enabled_tie_groups"] == list(TIES)
    assert server_services._derive_grid_preview(saved, loaded["study"])["full_candidate_count"] == 2
    for row in results:
        assert row.params["closeCountLong"] == row.params["closeCountShort"]
        assert row.params["tBandLongPct"] == row.params["tBandShortPct"]


def test_full_reduced_order_accessors_and_semantics():
    full = build_grid_v2_plan(strategy.load_config(), GridV2Settings(enabled_tie_groups=TIES))
    assert full.deduped_candidate_count == full.candidate_table.enumerated_candidate_count == 2800
    config = small_config()
    plain = build_grid_v2_plan(config)
    tied = build_grid_v2_plan(config, GridV2Settings(enabled_tie_groups=TIES))
    expected = [c.semantic_key for c in plain.candidates if c.params["closeCountLong"] == c.params["closeCountShort"] and c.params["tBandLongPct"] == c.params["tBandShortPct"]]
    assert [c.semantic_key for c in tied.candidates] == expected
    for i in range(len(tied.candidate_table)):
        table = tied.candidate_table
        p = table.params_for_index(i)
        for source, target in table.parameter_ties:
            assert table.has_param_for_index(i,target)
            assert table.param_value_for_index(i,target) == p[source] == p[target]


@pytest.mark.parametrize("budget", [1, 19, 32, 40])
def test_sampled_fixed_and_inactive_edits(budget):
    config = small_config()
    settings = GridV2Settings(enabled_tie_groups=TIES, planning_policy="sampled", requested_budget=budget)
    plan = build_grid_v2_plan(config, settings, {"closeCountShort":"stale", "tBandShortPct":float("nan")})
    other = build_grid_v2_plan(config, settings, {"closeCountShort":7, "tBandShortPct":2.})
    assert plan.plan_fingerprint == other.plan_fingerprint
    assert plan.deduped_candidate_count == min(budget, 32)
    for i in range(plan.deduped_candidate_count):
        p=plan.candidate_table.params_for_index(i)
        assert p["closeCountShort"] == plan.candidate_table.param_value_for_index(i,"closeCountLong")
        assert p["tBandLongPct"] == p["tBandShortPct"]
    fixed = build_grid_v2_plan(config, replace(settings, enabled_axes=()))
    assert fixed.deduped_candidate_count == 1
    assert fixed.candidates[0].params["closeCountShort"] == 7
    with pytest.raises(ValueError, match="shared source"):
        build_grid_v2_plan(config, replace(settings, enabled_axes=("closeCountShort",)))


def test_sampled_does_not_enumerate_and_seed_changes_membership(monkeypatch):
    import core.grid_v2 as grid
    def forbidden(*args,**kwargs): raise AssertionError("Sampled K<N called full builder")
    monkeypatch.setattr(grid,"_build_candidate_table",forbidden)
    config=strategy.load_config()
    settings=GridV2Settings(enabled_tie_groups=TIES,planning_policy="sampled",requested_budget=37)
    a=build_grid_v2_plan(config,settings)
    b=build_grid_v2_plan(config,settings)
    c=build_grid_v2_plan(config,replace(settings,seed=43))
    assert a.candidate_table.semantic_keys_by_row == b.candidate_table.semantic_keys_by_row
    assert a.candidate_table.semantic_keys_by_row != c.candidate_table.semantic_keys_by_row
    assert len(set(a.candidate_table.semantic_keys_by_row)) == 37


def test_sampled_disjointness_does_not_use_derived_target_seeds():
    import core.grid_v2 as grid
    prelude=grid._grid_v2_plan_prelude(small_config(),GridV2Settings(enabled_tie_groups=TIES),{})
    a=prelude.blocks[0]
    b=replace(a,name="other",seed_params={**a.seed_params,"closeCountShort":99})
    with pytest.raises(ValueError,match="disjointness"):
        grid._prove_sampled_blocks_semantically_disjoint(prelude.profile,[a,b])
    # A genuinely fixed active source can still distinguish two blocks.
    a=replace(a,axis_names=tuple(n for n in a.axis_names if n != "closeCountLong"))
    b=replace(a,name="other",seed_params={**a.seed_params,"closeCountLong":3})
    grid._prove_sampled_blocks_semantically_disjoint(prelude.profile,[a,b])


def test_selected_compiled_rows_have_one_authoritative_slow_call(monkeypatch):
    import core.grid_v2 as grid
    calls=[]
    original=grid.run_v2_strategy
    def count(**kwargs):
        calls.append(dict(kwargs["params"]))
        return original(**kwargs)
    monkeypatch.setattr(grid,"run_v2_strategy",count)
    plan=build_grid_v2_plan(small_config(),GridV2Settings(enabled_tie_groups=TIES,top_n=2))
    result=execute_grid_v2_candidates(plan,frame(400),100,GridV2StrategyHooks.from_strategy(strategy))
    assert result.metadata["compiled_batch_used"]
    assert len(calls) == len(result.selected) == 2
    for selected,params in zip(result.selected,calls):
        assert selected.row.semantic_key == plan.candidate_for_id(selected.row.candidate_id).semantic_key
        assert params["closeCountLong"] == params["closeCountShort"]


@pytest.mark.parametrize("emergency", [False, True])
@pytest.mark.slow
def test_compiled_reference_workers_and_selected(emergency):
    config = small_config()
    df = frame(1400)
    settings = GridV2Settings(enabled_tie_groups=TIES, top_n=2)
    hooks = GridV2StrategyHooks.from_strategy(strategy)
    runs=[]
    for compiled, workers in [(False,1),(True,1),(True,2)]:
        plan=build_grid_v2_plan(config,replace(settings,prefer_compiled=compiled,compiled_workers=workers), {"useEmergencySL":emergency})
        runs.append(execute_grid_v2_candidates(plan,df,100,hooks,compute_sharpe=True,compute_sharpe_daily=True,compute_sqn=True))
        if compiled:
            assert runs[-1].metadata["backend_kind"] == COMPILED_BATCH_KIND
            assert runs[-1].metadata["compiled_batch_used"] is True
    for run in runs[1:]:
        for actual, expected in zip(run.rows,runs[0].rows):
            for key, value in asdict(expected).items():
                observed=getattr(actual,key)
                if isinstance(value,float):
                    if math.isnan(value): assert math.isnan(observed)
                    else: assert observed == pytest.approx(value,rel=1e-9,abs=1e-10),key
                elif key == "backend_kind":
                    assert observed == "compiled_numba" and value == "reference"
                elif key == "guardrail_summary":
                    assert set(observed) <= set(value)
                    assert all(value[k] == v for k,v in observed.items())
                else: assert observed == value,key
        assert len(run.selected) == 2


@pytest.mark.parametrize("policy", ["full", "sampled"])
def test_value_cache_identity_and_rebase(policy):
    import core.grid_v2 as grid
    config = small_config()
    hooks = GridV2StrategyHooks.from_strategy(strategy)
    context = grid._cache_key_context(frame(), 100, hooks)
    settings = GridV2Settings(enabled_tie_groups=TIES, planning_policy=policy, requested_budget=17)
    cache = GridV2PlanReuseCache()
    first = cache.get_or_build(config, settings=settings)
    first.plan.candidate_table.params_for_index(0)
    second = cache.get_or_build(config, settings=settings, base_params={"end":"2025-02-01T00:00Z", "closeCountShort":"stale"})
    assert second.hit and second.plan.plan_fingerprint == first.plan.plan_fingerprint
    assert second.plan.candidates[0].params["end"] == "2025-02-01T00:00Z"
    untied = build_grid_v2_plan(config)
    keys = {c.semantic_key: i for i,c in enumerate(untied.candidates)}
    for i,c in enumerate(second.plan.candidates):
        for signal_only in (True,False):
            assert grid._cache_key_payload_for_index(second.plan,i,context,hooks,signal_only=signal_only) == grid._cache_key_payload_for_index(untied,keys[c.semantic_key],context,hooks,signal_only=signal_only)
        assert c.params["closeCountLong"] == c.params["closeCountShort"]
    assert not cache.get_or_build(config, settings=replace(settings, enabled_tie_groups=())).hit


@pytest.mark.parametrize("change", ["overlap", "unknown", "bool", "execution", "dependency", "mismatch", "malformed", "duplicate"])
def test_invalid_declaration(change):
    config = strategy.load_config()
    groups = config["optimization_rules"]["parameter_tie_groups"]
    if change == "overlap": groups[0]["pairs"].append(["closeCountLong","maLength3"])
    if change == "unknown": groups[0]["pairs"][0][1]="missing"
    if change == "bool": config["parameters"]["closeCountLong"]["type"]="bool"
    if change == "execution": config["parameters"]["closeCountLong"]["role"]="execution"
    if change == "dependency": config["parameters"]["closeCountLong"]["depends_on"]="enableLong"
    if change == "mismatch": config["parameters"]["closeCountLong"]["max"]=99
    if change == "malformed": groups[0]["pairs"]="bad"
    if change == "duplicate": groups.append(dict(groups[0]))
    with pytest.raises(ValueError): preview_grid_v2_counts(config)


@pytest.mark.parametrize("end,expected", [("2025-01-02",96),("2025-01-02T00:00Z",49),("2025-01-02T08:00+08:00",49),("2025-01-02T00:15Z",49),("2024-12-31T00:00Z",0)])
def test_end_single_batch_and_public(end,expected):
    df = frame(120)
    params = strategy.normalized_params({"end":end, "maType3":"SuperSmoother", "maLength3":25})
    single = strategy.build_v2_execution_data(df,params)
    batch = strategy.build_v2_execution_data_batch(df,[params, {**params,"dateFilter":False}])
    assert len(single.close) == len(batch[0].close) == expected
    assert len(batch[1].close) == len(df)
    result = strategy.S03ReversalV164AAdaptiveMAB2.run(df,params)
    assert len(result.equity_curve) == expected
    if result.trades: assert result.trades[-1].exit_time <= single.timestamps[-1]


def test_batch_reuses_ma_and_all_signal_fields_change(monkeypatch):
    df = frame(1400)
    original = signals.moving_average
    calls=[]
    def count(*args):
        calls.append(args[1:])
        return original(*args)
    monkeypatch.setattr(signals,"moving_average",count)
    rows=[strategy.normalized_params({"maType3":"KAMA", "closeCountLong":n}) for n in range(1,8)]
    strategy.build_v2_execution_data_batch(df,rows)
    assert calls == [("KAMA",75)]
    base = {"maType3":"SuperSmoother", "maLength3":25, "closeCountLong":1,"closeCountShort":1,"tBandLongPct":.2,"tBandShortPct":.2}
    original_data = strategy.build_v2_execution_data(df,base)
    for name,value in {"maType3":"KAMA","maLength3":250,"closeCountLong":7,"closeCountShort":7,"tBandLongPct":2.,"tBandShortPct":2.}.items():
        other=strategy.build_v2_execution_data(df,{**base,name:value})
        assert np.any(original_data.signals.long_entries != other.signals.long_entries) or np.any(original_data.signals.short_entries != other.signals.short_entries),name


@pytest.mark.slow
def test_chunked_population_matches_unchunked():
    config = small_config()
    df = frame(1400)
    hooks=GridV2StrategyHooks.from_strategy(strategy)
    settings=GridV2Settings(enabled_tie_groups=TIES,slow_enrich_selected=False)
    plan=build_grid_v2_plan(config,settings)
    normal=execute_grid_v2_candidates(plan,df,100,hooks)
    chunked=execute_grid_v2_candidates(replace(plan,settings=replace(settings,max_signal_cache_mb=.12)),df,100,hooks)
    assert chunked.metadata["chunk_count"] > 1
    assert [row.semantic_key for row in chunked.rows] == [row.semantic_key for row in normal.rows]
    assert [row.net_profit_pct for row in chunked.rows] == [row.net_profit_pct for row in normal.rows]
    assert chunked.cache_estimate.signal_combo_count == 32


@pytest.mark.parametrize("mode",["days","months","adaptive"])
@pytest.mark.slow
def test_wfa_reuse_storage_and_explicit_replay(mode):
    from core import storage
    from core.walkforward_engine import WFConfig, WalkForwardEngine
    df=frame(480)
    if mode == "months": df.index=pd.date_range("2025-01-01",periods=len(df),freq="6h",tz="UTC")
    fixed=strategy.normalized_params({"maType3":"SuperSmoother","maLength3":25,"dateFilter":mode == "months","start":df.index[0].isoformat(),"end":df.index[-1].isoformat()})
    fixed.pop("warmupBars")
    template={"strategy_id":strategy.S03ReversalV164AAdaptiveMAB2.STRATEGY_ID,"optimization_mode":"grid",
              "enabled_params":{"closeCountLong":True},"param_ranges":{"closeCountLong":[1,2,1]},"param_types":{"closeCountLong":"int"},
              "fixed_params":fixed,"grid_v2_enabled_tie_groups":list(TIES),"objectives":["net_profit_pct"],"grid_fast_objectives":["net_profit_pct"],
              "grid_top_candidates":1,"grid_diversity_enabled":False,"worker_processes":1,"risk_per_trade_pct":2.,"contract_size":.01,"commission_rate":.0005,"filter_min_profit":False,"min_profit_threshold":0.}
    wf=WFConfig(strategy_id=template["strategy_id"],is_period_days=4,oos_period_days=2,warmup_bars=100,adaptive_mode=mode=="adaptive",max_oos_period_days=2)
    if mode == "months": wf=WFConfig(strategy_id=template["strategy_id"],period_unit="months",is_period_days=None,oos_period_days=None,is_period_months=1,oos_period_months=1,warmup_bars=100)
    engine=WalkForwardEngine(wf,template,{},csv_file_path="task-owned-s03.csv")
    result,study_id=engine.run_wf_optimization(df)
    assert result.windows and study_id
    saved=storage.load_study_from_db(study_id)["study"]["config_json"]
    assert saved["grid_v2_enabled_tie_groups"] == saved["grid_config"]["enabled_tie_groups"] == list(TIES)
    assert engine._grid_v2_plan_cache.stats.build_count == 1
    for window in result.windows:
        assert window.best_params["closeCountLong"] == window.best_params["closeCountShort"]
        assert window.module_status["grid_v2"]["candidate_count"] == 2
    asymmetric={**fixed,"closeCountLong":4,"closeCountShort":6,"dateFilter":False}
    a=strategy.S03ReversalV164AAdaptiveMAB2.run(df,asymmetric,100)
    b=strategy.S03ReversalV164AAdaptiveMAB2.run(df,{**asymmetric,"grid_v2_enabled_tie_groups":list(TIES)},100)
    assert a.trades == b.trades
    # Delayed OOS uses explicit selected parameters and strips technical history.
    live_start, scheduled_start = df.index[150], df.index[120]
    replay = engine._run_period_backtest(df, live_start, df.index[-1], asymmetric)
    delayed = engine._prepend_flat_prefix(replay, scheduled_start=scheduled_start, live_start=live_start)
    assert delayed.trades == replay.trades
    assert delayed.timestamps[0] == scheduled_start
    assert all(t >= live_start for t in delayed.timestamps[1:])
    assert delayed.metric_start_idx == 0
    assert all(t.entry_time >= live_start for t in delayed.trades)
