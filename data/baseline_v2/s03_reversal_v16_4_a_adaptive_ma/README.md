# S03 Reversal v16-4-A Adaptive MA TradingView Baselines

**Status:** Complete and internally verified  
**Purpose:** External evidence for the Backtester V2 import  
**Proposed strategy ID:** `s03_reversal_v16_4_a_adaptive_ma_b2`

This package contains four TradingView references, one for each adaptive moving
average offered by the Pine strategy. Two references keep Emergency SL disabled;
two enable it at 10% and collectively exercise Long and Short stop exits. Raw
TradingView exports and screenshots are preserved unchanged. Machine-readable
UTC trades, parameters, summaries, and the root `dataset.json` manifest are
derived from those immutable inputs.

## Frozen source and market

- Pine authority: `pine/S_03-Reversal_v16-4-A_Adaptive-MA.pine`
  (`aa45babe2ee8b6a8a61250a217c116346ad6d9e7b55dc9a3b81f23edc5bab47c`).
- TradingView symbol: `OKX:SUIUSDT.P`; timeframe: `30m`; UI/export timezone:
  `UTC+8`.
- UTC interval: `[2025-08-01T00:00:00Z, 2025-12-01T00:00:00Z)`.
- Merlin market data: `data/raw/OKX_SUIUSDT.P, 30 2025.01.01-2026.02.01.csv`
  (`d664bbae2903828f84b19e7af548fdc744b970a17f56846ad77882a9ca786aae`).
- Tick size: `0.0001`.

## Common execution profile

All references use both Long and Short, contract size `0.01`, initial capital
`100 USDT`, 100% of equity sizing rounded down from the signal close, pyramiding
`0`, commission `0.05%`, slippage `0`, default four-tick bar detailization,
standard OHLC fills, Bar Magnifier off, process-on-close off, order-fill and
realtime-tick recalculation off, and one-tick order delay.

Normal entries and signal closes fill at the next bar open. A reversal first
closes the opposite position and submits a new entry only after the strategy is
flat. Date Filter uses an inclusive Start and exclusive End; every reference
exercises `End Close` immediately on the boundary bar's closing tick.

## Accepted references

| Reference | MA / length | Close L/S | T Band L/S | Emergency SL | Trades | Net PnL | PF |
|---|---:|---:|---:|---:|---:|---:|---:|
| [A: KAMA](reference_a_kama/README.md) | `KAMA` / `125` | `4/6` | `1.0/1.2%` | Off | 63 | 43.65% | 1.36 |
| [B: SuperSmoother](reference_b_supersmoother/README.md) | `SuperSmoother` / `175` | `7/4` | `1.8/0.8%` | Off | 66 | 44.39% | 1.297 |
| [C: FRAMA + Emergency SL 10%](reference_c_frama+sl10/README.md) | `FRAMA` / `100` | `6/6` | `1.8/1.8%` | 10% (3 exits: 1 L / 2 S) | 62 | 2.39% | 1.017 |
| [D: DSMA + Emergency SL 10%](reference_d_dsma+sl10/README.md) | `DSMA` / `200` | `6/7` | `1.6/1.8%` | 10% (3 exits: 0 L / 3 S) | 53 | 14.04% | 1.133 |

## Verification facts

- Every raw export has exactly one Entry and one Exit row for every sequential
  trade number.
- Trade counts, profitable counts, final cumulative PnL, and displayed Profit
  Factor agree with the corresponding metrics screenshot.
- All 488 raw rows map to the local UTC market data after subtracting eight
  hours. Normal market fills equal the matching bar Open; each End Close equals
  the boundary bar Close; every Emergency SL price lies inside its bar's OHLC.
- The two Emergency SL references contain six actual stop exits: one Long and
  five Short.
- Sums of individually rounded two-decimal PnL rows can differ by a few cents
  from TradingView's higher-precision cumulative value. The screenshot and
  final cumulative field are authoritative.

## Timezone and boundary interpretation

TradingView screenshots and raw exports use `UTC+8`; all generated timestamps
use UTC. TradingView labels the immediate final close with the 30-minute
boundary bar's opening timestamp, `2025-12-01 07:30 UTC+8`
(`2025-11-30T23:30:00Z`). Its semantic close boundary is
`2025-12-01 08:00 UTC+8` (`2025-12-01T00:00:00Z`).

## Symmetric Grid interpretation

`Use Symmetric Long/Short Parameters` is planned as a Merlin Grid-planning
constraint, not a Pine input. It restricts candidate construction by pairing
equal Close Count and T Band values. It must not alter the execution or identity
of an already expanded candidate containing explicit Long and Short values.
These external references therefore record only the explicit Pine parameters.

## File roles

- `pine/`: immutable Pine source used for all TradingView references.
- `tradingview_trades.csv`: immutable raw TradingView export in UTC+8.
- `trades_normalized_utc.csv`: one row per closed trade with UTC timestamps.
- `tradingview_inputs.PNG`, `tradingview_properties.PNG`, and
  `tradingview_metrics.PNG`: immutable UI evidence.
- `params.json`: machine-readable inputs and execution semantics.
- `tradingview_summary.json`: transcribed metrics and raw-export consistency.
- Each reference directory has a concise local README.

## Separate Merlin import interpretation

The production import is `s03_reversal_v16_4_a_adaptive_ma_b2`. It intentionally
uses the existing shared Merlin KAMA initialization and includes a bar whose
opening timestamp equals runtime End. The four frozen TradingView references
above retain their original meaning and files. Separate production metrics,
quantities, PnL and exact price residuals are in
[merlin_expectations.json](merlin_expectations.json); the explanatory authority
is [V2 certification](../../../docs/engine_v2/CERTIFICATION.md#s03-v16-4-a-adaptive-ma-import).
Reference D's additional final bar executes a Long Emergency SL at 1.42443,
giving four production stop exits versus the export's three. Six non-final
stop fills have bounded exported-price residuals; they are enumerated in the
certification entry, including three omitted by the preparation specification.
