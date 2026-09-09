"""Read-only S03 reference characterization and bounded Grid performance run.

Run with the project Python and --output pointing outside every checkout.
This tool never writes the frozen baselines or operational storage.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.is_relative_to(root) or any((p / ".git").exists() for p in (output, *output.parents)):
        parser.error("Output must be outside every checkout.")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(output / "numba"))
    cache_preexisting = any(Path(os.environ["NUMBA_CACHE_DIR"]).rglob("*.nbc"))
    sys.pycache_prefix = str(output / "pycache")
    sys.path.insert(0, str(root / "src"))

    import numba
    import numpy as np
    import pandas as pd
    from core.backtest_engine import load_data, prepare_dataset_with_warmup
    from core.engine_v2.runner import run_v2_strategy
    from core.grid_v2 import GridV2Settings, GridV2StrategyHooks, build_grid_v2_plan, preview_grid_v2_counts, execute_grid_v2_candidates
    import core.grid_v2 as grid
    from strategies.s03_reversal_v16_4_a_adaptive_ma_b2 import strategy

    baseline = root / "data/baseline_v2/s03_reversal_v16_4_a_adaptive_ma"
    manifest = json.loads((baseline / "dataset.json").read_text())
    assets = [*manifest["asset_hashes"], manifest["pine_source"], manifest["market_data"]]
    for asset in assets:
        assert hashlib.sha256((root / asset["path"]).read_bytes()).hexdigest() == asset["sha256"], asset["path"]
    market = load_data(root / manifest["market_data"]["path"])
    start, end = pd.Timestamp("2025-08-01T00:00Z"), pd.Timestamp("2025-12-01T00:00Z")
    df, trade_start = prepare_dataset_with_warmup(market, start, end, 1000)
    runtime = {"dateFilter":True, "start":start.isoformat(), "end":end.isoformat()}
    evidence = {"runtime":runtime, "warmup_bars":1000, "bars":len(df), "trade_start_idx":trade_start, "verified_hashes":len(assets), "references":{}}
    for ref in manifest["references"]:
        path = root / ref["path"]
        params = strategy.normalized_params({**json.loads((path / "params.json").read_text())["strategy_inputs"], **runtime})
        result = run_v2_strategy(data=strategy.build_v2_execution_data(df,params),profile=strategy.load_profile(),params=params,trade_start_idx=trade_start)
        tv = list(csv.DictReader((path / "trades_normalized_utc.csv").open(encoding="utf-8-sig")))
        rows=[]
        for number,(trade,external) in enumerate(zip(result.strategy_result.trades,tv),1):
            assert trade.direction == external["direction"]
            assert trade.entry_time == pd.Timestamp(external["entry_time_utc"])
            assert trade.entry_price == float(external["entry_price_usdt"])
            # Independently reconstruct this trade's net PnL with the frozen commission convention.
            sign = 1 if trade.direction == "long" else -1
            net = trade.size*(sign*(trade.exit_price-trade.entry_price) - (trade.entry_price+trade.exit_price)*.0005)
            assert abs(net-trade.net_pnl) < 1e-10
            rows.append({"number":number, **asdict(trade),
                         "tv_exit_time":external["exit_time_utc"], "tv_exit_price":float(external["exit_price_usdt"]),
                         "size_delta":trade.size-float(external["size_qty"]), "net_pnl_delta":trade.net_pnl-float(external["net_pnl_usdt"]),
                         "exit_price_delta":trade.exit_price-float(external["exit_price_usdt"])})
        convergence={}
        for warmup in (100,500,1000,2000):
            prepared, index = prepare_dataset_with_warmup(market,start,end,warmup)
            average = strategy.build_v2_execution_data(prepared,params)
            canonical = strategy.build_v2_execution_data(df,params)
            differences = np.count_nonzero(average.signals.long_entries[index:] != canonical.signals.long_entries[trade_start:]) + np.count_nonzero(average.signals.short_entries[index:] != canonical.signals.short_entries[trade_start:])
            convergence[str(warmup)] = int(differences)
        evidence["references"][ref["reference_id"]]={"params":params,"basic":asdict(result.basic_metrics),"advanced":asdict(result.advanced_metrics),"trades":rows,"signal_differences_from_warmup_1000":convergence}
    (output / "references.json").write_text(json.dumps(evidence,indent=2,default=str)+"\n",encoding="utf-8")
    report={"host":platform.platform(),"processor":platform.processor(),"logical_cpus":os.cpu_count(),"python":platform.python_version(),"numpy":np.__version__,"numba":numba.__version__,"numba_threads":numba.get_num_threads(),"bars":len(df),"trade_start_idx":trade_start,"workers":1,"cache_mb":32,"seed":42,"runs":[]}
    config=strategy.load_config()
    hooks=GridV2StrategyHooks.from_strategy(strategy)
    slow_times=[]
    original_enrich=grid._slow_enrich_selected
    def timed_enrich(*args,**kwargs):
        started=time.perf_counter()
        result=original_enrich(*args,**kwargs)
        slow_times.append(time.perf_counter()-started)
        return result
    grid._slow_enrich_selected=timed_enrich
    for ties in ((),("symmetricLongShort",)):
        settings=GridV2Settings(enabled_tie_groups=ties,max_signal_cache_mb=32,compiled_workers=1,top_n=2)
        started=time.perf_counter(); preview=preview_grid_v2_counts(config,settings,runtime)
        report.setdefault("preview",[]).append({"ties":ties,"count":preview.raw_candidate_count,"seconds":time.perf_counter()-started,"details":asdict(preview)})
    for kind in ("symmetric_full","asymmetric_sampled"):
        settings=GridV2Settings(enabled_tie_groups=("symmetricLongShort",) if kind == "symmetric_full" else (), planning_policy="full" if kind == "symmetric_full" else "sampled",requested_budget=128,max_signal_cache_mb=32,compiled_workers=1,top_n=2)
        started=time.perf_counter(); plan=build_grid_v2_plan(config,settings,runtime); planning_seconds=time.perf_counter()-started
        assert {plan.candidate_table.param_value_for_index(i,"maType3") for i in range(plan.deduped_candidate_count)} == {"KAMA","SuperSmoother","FRAMA","DSMA"}
        for iteration in range(2):
            slow_times.clear()
            started=time.perf_counter(); result=execute_grid_v2_candidates(plan,df,trade_start,hooks,compute_sharpe=True,compute_sqn=True,compute_sharpe_daily=True)
            seconds=time.perf_counter()-started
            assert result.metadata["compiled_batch_used"]
            assert not any(row.error for row in result.rows)
            report["runs"].append({"kind":kind,"iteration":iteration,"jit":("disk cache present" if cache_preexisting else "cold compiled cache") if kind == "symmetric_full" and iteration == 0 else "warm","candidate_count":plan.deduped_candidate_count,"planning_seconds":planning_seconds,"execution_wall_seconds":seconds,"selected_slow_seconds":sum(slow_times),"plan_fingerprint":plan.plan_fingerprint,"planning_metadata":dict(plan.metadata),"per_block_counts":dict(plan.per_block_counts),"settings":asdict(settings),"cache_estimate":asdict(result.cache_estimate),"metadata":dict(result.metadata),"selected":[{"id":x.row.candidate_id,"semantic_key":x.row.semantic_key,"metrics":x.metrics} for x in result.selected]})
            print(kind,iteration,plan.deduped_candidate_count,round(seconds,4),result.metadata["backend_kind"],flush=True)
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class MemoryCounters(ctypes.Structure):
            _fields_ = [("cb",wintypes.DWORD),("faults",wintypes.DWORD)] + [(name,ctypes.c_size_t) for name in ("peak_working_set","working_set","peak_paged_pool","paged_pool","peak_nonpaged_pool","nonpaged_pool","pagefile","peak_pagefile")]
        counters=MemoryCounters()
        counters.cb=ctypes.sizeof(counters)
        query=ctypes.WinDLL("psapi",use_last_error=True).GetProcessMemoryInfo
        query.argtypes=[wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD]
        query.restype=wintypes.BOOL
        if not query(wintypes.HANDLE(-1),ctypes.byref(counters),counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        report["peak_process_working_set_bytes"]=counters.peak_working_set
    else:
        import resource
        report["peak_process_working_set_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    (output / "performance.json").write_text(json.dumps(report,indent=2,default=str)+"\n",encoding="utf-8")
    print(output,flush=True)


if __name__ == "__main__":
    main()
