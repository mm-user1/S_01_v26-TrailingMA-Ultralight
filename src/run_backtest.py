import argparse
import pandas as pd

from backtest_engine import StrategyParams, load_data, prepare_dataset_with_warmup, run_strategy_v2
from strategy_registry import StrategyRegistry

DEFAULT_CSV_PATH = "../data/OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv"


def _parse_date(value):
    if value in (None, ""):
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strategy backtest")
    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_CSV_PATH,
        help="Path to the CSV file with OHLCV data",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="s01_trailing_ma",
        help="Strategy ID to run",
    )
    args = parser.parse_args()

    df = load_data(args.csv)

    strategy_class = StrategyRegistry.get_strategy_class(args.strategy)
    param_defs = strategy_class.get_param_definitions()
    params_dict = {key: value["default"] for key, value in param_defs.items()}

    # Warmup handling uses StrategyParams for compatibility
    start = _parse_date(params_dict.get("startDate"))
    end = _parse_date(params_dict.get("endDate"))
    params_obj = StrategyParams(
        use_backtester=params_dict.get("useBacktester", True),
        use_date_filter=params_dict.get("dateFilter", True),
        start=start,
        end=end,
        ma_type=str(params_dict.get("maType", "EMA")).upper(),
        ma_length=int(params_dict.get("maLength", 0)),
        close_count_long=int(params_dict.get("closeCountLong", 0)),
        close_count_short=int(params_dict.get("closeCountShort", 0)),
        stop_long_atr=float(params_dict.get("stopLongAtr", 0.0)),
        stop_long_rr=float(params_dict.get("stopLongRr", 0.0)),
        stop_long_lp=int(params_dict.get("stopLongLp", 0)),
        stop_short_atr=float(params_dict.get("stopShortAtr", 0.0)),
        stop_short_rr=float(params_dict.get("stopShortRr", 0.0)),
        stop_short_lp=int(params_dict.get("stopShortLp", 0)),
        stop_long_max_pct=float(params_dict.get("stopLongMaxPct", 0.0)),
        stop_short_max_pct=float(params_dict.get("stopShortMaxPct", 0.0)),
        stop_long_max_days=int(params_dict.get("stopLongMaxDays", 0)),
        stop_short_max_days=int(params_dict.get("stopShortMaxDays", 0)),
        trail_rr_long=float(params_dict.get("trailRrLong", 0.0)),
        trail_rr_short=float(params_dict.get("trailRrShort", 0.0)),
        trail_ma_long_type=str(params_dict.get("trailMaLongType", "SMA")).upper(),
        trail_ma_long_length=int(params_dict.get("trailMaLongLength", 0)),
        trail_ma_long_offset=float(params_dict.get("trailMaLongOffset", 0.0)),
        trail_ma_short_type=str(params_dict.get("trailMaShortType", "SMA")).upper(),
        trail_ma_short_length=int(params_dict.get("trailMaShortLength", 0)),
        trail_ma_short_offset=float(params_dict.get("trailMaShortOffset", 0.0)),
        risk_per_trade_pct=float(params_dict.get("riskPerTradePct", 0.0)),
        contract_size=float(params_dict.get("contractSize", 0.0)),
        commission_rate=float(params_dict.get("commissionRate", 0.0)),
        atr_period=int(params_dict.get("atrPeriod", 0)),
    )

    if params_obj.use_date_filter and (params_obj.start is not None or params_obj.end is not None):
        df_prepared, trade_start_idx = prepare_dataset_with_warmup(
            df, params_obj.start, params_obj.end, params_obj
        )
    else:
        df_prepared = df
        trade_start_idx = 0

    params_dict["startDate"] = start
    params_dict["endDate"] = end
    strategy = strategy_class(params_dict)
    result = run_strategy_v2(df_prepared, strategy, trade_start_idx)

    print(f"Net Profit %: {result.net_profit_pct:.2f}")
    print(f"Max Portfolio Drawdown %: {result.max_drawdown_pct:.2f}")
    print(f"Total Trades: {result.total_trades}")


if __name__ == "__main__":
    main()
