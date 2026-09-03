# Merlin Metrics

This is the cross-engine authority for current metric behavior shared by
Backtester V1, Backtester V2, Grid, WFA, storage/UI, and Strategy Lab. Exact
certification values and external residuals belong in
[V2 certification](engine_v2/CERTIFICATION.md), not here.

## Ownership and categories

`src/core/metrics.py` owns canonical `BasicMetrics`, `AdvancedMetrics`, and
`WFAMetrics` calculations. Strategy execution produces trades plus balance and
equity curves; metric code interprets those outputs. Fast Grid implementations
must match the canonical reference for every metric they advertise.

- Realized/basic metrics use completed trades and the realized balance curve.
- Advanced path metrics use the explicitly bounded mark-to-market equity view.
- WFA reports per-window IS/OOS metrics and aggregate/stitched facts.
- Strategy Lab projects a fixed research schema from certified V2 execution;
  its MTM drawdown is not a public Merlin optimization metric.

## Evaluation boundary

`StrategyResult.metric_start_idx` is the first evaluation observation in the
curves. `metric_initial_equity` is the equity immediately before it. Producers
must set both when prepared data includes technical warmup.

Monthly Sharpe, Monthly Sortino, Daily Sharpe, Ulcer Index, and Consistency use
the evaluation-only equity observations and their pre-evaluation anchor. The
boundary excludes technical warmup without inventing observations. Realized
basic metrics—including realized Max DD and RoMaD—do not use this advanced
boundary.

## Realized metrics

Net Profit, gross profit/loss, Profit Factor, trade counts, Win Rate, average
trade statistics, and maximum consecutive losses are derived from completed
trades. Percentage Net Profit uses the supplied initial balance where the
calling surface provides it.

Realized Max DD is balance-based. It scans every finite value in the complete
realized `balance_curve`, including its final observation and any flat
technical-warmup prefix. At each observation the running realized-balance peak
is updated. Percentage drawdown is the maximum
`(peak - balance) / peak * 100` for positive peaks; absolute drawdown is the
independent maximum `peak - balance`. Neither `metric_start_idx`,
`metric_initial_equity`, nor a Net Profit starting-balance argument seeds or
truncates this scan. An unrecovered final loss therefore remains visible.

RoMaD uses Net Profit percentage divided by this realized percentage Max DD.
It retains the established zero-drawdown convention in `calculate_advanced`;
it must not be recomputed from MTM drawdown.

## Monthly risk-adjusted metrics

Monthly returns are built from evaluation-only bar-close equity. The
pre-evaluation anchor opens the first month, a transition bar belongs to the
new UTC calendar month, the preceding bar closes the old month, and the final
partial month is retained. Technical warmup months are not observations.

Monthly Sharpe (`sharpe_ratio`) uses a fixed 2% annual risk-free rate divided
by 12 and population variance. It is not annualized with a square root. It is
undefined without completed trades, with fewer than two real monthly returns,
or with non-positive/non-finite variance.

Monthly Sortino (`sortino_ratio`) uses the same monthly series and requires
real downside observations. Two- to four-month windows commonly yield `None`;
selecting it as a Slow objective can remove many or all candidates.

## Daily Sharpe

`sharpe_daily` is distinct from Monthly Sharpe and is calculated only when
requested. Evaluation timestamps are grouped into observed UTC days since the
Unix epoch. The algorithm uses the pre-evaluation equity anchor, fractional
simple returns for observed days (including partial edge days), `0.02 / 365`
as the daily risk-free rate, population variance, and `sqrt(365)`
annualization. Missing calendar dates are not synthesized.

Any non-finite evaluation equity observation or non-positive opening
denominator invalidates the complete daily series. When disabled or
structurally invalid, `sharpe_daily_observations` and
`sharpe_daily_active_days` are `None`. For a valid constructed series both are
integers, including zero; the ratio can still be `None` with no trades, too few
observations, or zero variance. An active day has absolute raw return greater
than `1e-12`; this is diagnostic, not an eligibility gate.

The reference implementation and V1/V2 Fast implementations share these
boundaries and validity rules. Fast execution uses one dataset/window UTC-day
array and per-candidate streaming statistics. Fixed and Adaptive WFA final
reporting requests Daily Sharpe for real IS and real undelayed dense OOS
series. Delayed/no-trade sparse OOS does not publish it because missing dates
must not be invented. Storage fields are nullable and historical rows are not
backfilled.

## Other advanced metrics

- Profit Factor is gross winning PnL divided by absolute gross losing PnL when
  defined.
- SQN uses exact net trade PnL and sample variance. It is undefined below 30
  completed trades, with non-positive variance, or standard deviation below
  `1e-10`.
- Ulcer Index measures drawdown along the advanced evaluation equity path.
- Consistency is signed R² of that evaluation equity path, in `[-1, 1]`.
- Composite Score is an Optuna/UI scoring construct over configured normalized
  metrics, not an independently generated market-performance series.

Non-finite selected objective values are not silently replaced. Optuna marks
invalid objective returns as failed trials; Grid removes candidates with a
non-finite selected objective from ranking. Short WFA windows can therefore
have no rankable SQN, Daily Sharpe, or Sortino candidate.

## Surface availability

`Yes` means the metric is available on that surface when its mathematical
preconditions hold. “Gated” means it is computed only when requested or during
the surface's explicit reporting path.

| Metric or group | V1 Optuna objective | Grid Fast | Grid Slow | WFA reporting | Strategy Lab schema v2 |
| --- | --- | --- | --- | --- | --- |
| Net Profit %, realized Max DD %, RoMaD, Profit Factor, Win Rate | Yes | V1 + V2 | Yes | Per-window/aggregate as applicable | Yes |
| Monthly Sharpe (`sharpe_ratio`) | Yes | Gated, V1 + V2 | Yes | Reported when defined | No |
| Daily Sharpe | Gated | Gated, V1 + V2; Fast-only | No | Gated final IS/dense OOS | Yes, with observation/active-day diagnostics |
| SQN | Yes | Gated, V1 + V2 | Yes | Reported when defined | Yes |
| Monthly Sortino | Yes | No | Yes | Reported when defined | No |
| Ulcer Index, Consistency | Yes | No | Yes | Reported when defined | No |
| Composite Score | Yes | No | No | Selection metadata where applicable | No |
| Total Trades, Max Consecutive Losses | Constraint/reporting; core direction exists | V2 Fast objective; V1 reporting/constraint | V2 Slow objective; V1 reporting/constraint | Yes | Yes |
| Winning/losing trades, gross profit/loss | Reporting | Fast result facts | Slow result facts | Per-window where carried | Yes |
| Guardrail diagnostics | No | V2 execution facts | V2 selected reference facts | Stored diagnostics where applicable | Yes |
| Bar-close MTM Max DD | No | Internal request-gated V2 sidecar only | No | No | Position-family only |

Grid permits at most six Fast objectives. The common V1/V2 Fast set is Net
Profit %, realized Max DD %, RoMaD, Profit Factor, Win Rate, Monthly Sharpe,
Daily Sharpe, and SQN. V2 additionally allows Total Trades and Max Consecutive
Losses. Slow refinement supports the common Fast set except Daily Sharpe and
adds Sortino, Ulcer Index, and Consistency; V2 also allows the two count/loss
metrics. Availability is validated by the current constants in
`src/core/grid_engine.py`.

V1 Optuna exposes one to six selections from Net Profit %, realized Max DD %,
Monthly Sharpe, Daily Sharpe, Sortino, RoMaD, Profit Factor, Win Rate, SQN,
Ulcer Index, Consistency, and Composite Score. Optuna remains V1-only for new
optimizer execution.

## Strategy Lab MTM drawdown

`max_drawdown_mtm_pct` is a request-gated research column for the generic V2
position family. It scans finite bar-close MTM equity from
`trade_start_idx` through the final bar and seeds the peak with positive
initial capital. An empty interval or any non-finite observation returns NaN;
a present flat interval returns `0.0`; negative equity can produce drawdown
above 100%. Signal-reversal requests fail explicitly.

This metric is not realized Max DD and does not alter the compiled result ABI;
compiled execution transports it in an optional sidecar. It is also not
TradingView intrabar/open-excursion drawdown, which can use within-bar High/Low
behavior unavailable to this bar-close series. A future `romad_mtm` would use
NaN when MTM drawdown is zero, unlike established realized RoMaD.

The Strategy Lab schema-v2 metric axis is authoritative in
`tools/strategy_lab/dataset.py`; its analysis and allocation interpretation is
documented in the [Strategy Lab guide](../tools/strategy_lab/README.md).
