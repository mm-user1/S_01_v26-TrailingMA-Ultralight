import argparse
import pandas as pd

from backtest_core import StrategyParams, load_data, run_strategy

DEFAULT_CSV_PATH = "OKX_LINKUSDT.P, 15 2025.02.01-2025.09.09.csv"


def build_default_params() -> StrategyParams:
    return StrategyParams(
        use_backtester=True,
        use_date_filter=True,
        start=pd.Timestamp("2025-04-01", tz="UTC"),
        end=pd.Timestamp("2025-09-01", tz="UTC"),
        ma_type="EMA",
        ma_length=45,
        close_count_long=7,
        close_count_short=5,
        stop_long_atr=2.0,
        stop_long_rr=3.0,
        stop_long_lp=2,
        stop_short_atr=2.0,
        stop_short_rr=3.0,
        stop_short_lp=2,
        stop_long_max_pct=3.0,
        stop_short_max_pct=3.0,
        stop_long_max_days=2,
        stop_short_max_days=4,
        trail_rr_long=1.0,
        trail_rr_short=1.0,
        trail_ma_long_type="SMA",
        trail_ma_long_length=160,
        trail_ma_long_offset=-1.0,
        trail_ma_short_type="SMA",
        trail_ma_short_length=160,
        trail_ma_short_offset=1.0,
        risk_per_trade_pct=2.0,
        contract_size=0.01,
        commission_rate=0.0005,
        atr_period=14,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the S_01 TrailingMA backtest")
    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_CSV_PATH,
        help="Path to the CSV file with OHLCV data",
    )
    args = parser.parse_args()

    df = load_data(args.csv)
    params = build_default_params()
    result = run_strategy(df, params)

    print(f"Net Profit %: {result.net_profit_pct:.2f}")
    print(f"Max Portfolio Drawdown %: {result.max_drawdown_pct:.2f}")
    print(f"Total Trades: {result.total_trades}")


if __name__ == "__main__":
    main()
