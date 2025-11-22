# Migration Prompt 7 Report

## What was done
- Added strategy selectors with metadata panels to the Backtester and Optimizer UI, wired to the `/api/strategies` endpoint.
- Implemented strategy-aware frontend state (strategy caching, change handlers), parameter visibility helpers, and mapping logic to align UI parameters with strategy definitions.
- Routed backtest/optimization requests with `strategy_id`, added reversal-aware result formatting, and ensured configs map parameters for each strategy.
- Updated CLAUDE.md with multi-strategy usage instructions.

## Reference tests
- `python run_backtest.py --csv "../data/OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv" --strategy s01_trailing_ma` → Net Profit 230.75%, Max DD 20.03%, Trades 93 (✅ matches baseline).
- `python run_backtest.py --csv "../data/OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv" --strategy s03_reversal` → Net Profit 83.56%, Max DD 35.34%, Trades 224 (✅ matches baseline).

## Notes
- Parameter visibility now keeps optimizer controls visible even when they do not directly map to strategy parameters, to avoid hiding shared optimization UI elements.
