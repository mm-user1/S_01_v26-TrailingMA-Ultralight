# S06 R-Trend v06-4-A2 TradingView Baselines

**Status:** Complete and internally verified  
**Purpose:** External evidence for the TZ-17 Backtester V2 import  
**Proposed strategy ID:** `s06_r_trend_v06_4_a2_b2`

This package contains six TradingView references for the three new stateful
trailing-stop modes, each in Reversal and Trend entry mode. Raw TradingView
exports and screenshots are preserved unchanged. Machine-readable UTC trades,
per-reference parameters and summaries, and the root `dataset.json` manifest
were derived from those immutable inputs.

## Frozen sources and market

- Pine authority:
  `pine/S_06-R-Trend_v06-4-A2_Fixed-AF-SAR.pine`
  (`2c832d4d0c6361a07ef4e3e7cc47960c50cbcd60524dbb100755369d17abbd6e`)
- Certification Pine:
  `pine/S_06-R-Trend_v06-4-A2_Fixed-AF-SAR_baseline-no-magnifier.pine`
  (`392098f6f52329c36ae846098e0d009613c268580c8e80f2a0e00f4e9dfa4d02`)
- The certification copy differs only by changing Bar Magnifier from enabled
  to disabled.
- TradingView symbol: `OKX:SUIUSDT.P`; timeframe: `30m`; UI/export timezone:
  `UTC+8`.
- UTC interval: `[2025-08-01T00:00:00Z, 2025-12-01T00:00:00Z)`.
- Merlin market data:
  `data/raw/OKX_SUIUSDT.P, 30 2025.01.01-2026.02.01.csv`
  (`d664bbae2903828f84b19e7af548fdc744b970a17f56846ad77882a9ca786aae`).
- Tick size: `0.0001`.

## Common inputs and execution profile

All references use both Long and Short, Fast `21/7`, Slow `112/3`, thresholds
`20/20`, Stop RR `2` (inactive in trail modes), risk `2%`, and contract size
`0.01`.

TradingView execution uses initial capital `100 USDT`, order size `100 USDT`
cash, pyramiding `0`, commission `0.05%`, slippage `0`, default four-tick bar
detailization, standard OHLC fills, Bar Magnifier off, process-on-close off,
calculate-on-order-fill off, calculate-on-every-tick off, and one-tick order
delay.

## Accepted references

| Reference | Entry / trail | Stop X / LP / Max % / Days | Active trail parameters | Trades | Profitable method exits |
|---|---|---:|---|---:|---:|
| A | Reversal / R Trail | `2 / 2 / 4 / 6` | activation `1R`, distance `2R` | 52 | 8 (4 long, 4 short) |
| B | Trend / R Trail | `2.5 / 2 / 8 / 4` | activation `1R`, distance `2.5R` | 42 | 3 (2 long, 1 short) |
| C | Reversal / Chandelier | `1.5 / 4 / 8 / 4` | activation `1R`, ATR `28`, multiplier `2` | 68 | 35 (19 long, 16 short) |
| D | Trend / Chandelier | `2.5 / 4 / 8 / 4` | activation `1R`, ATR `28`, multiplier `2` | 47 | 25 (10 long, 15 short) |
| E | Reversal / Fixed-AF SAR | `1.5 / 4 / 8 / 4` | activation `1R`, speed `0.005` | 51 | 18 (15 long, 3 short) |
| F | Trend / Fixed-AF SAR | `2.5 / 4 / 8 / 4` | activation `2R`, speed `0.005` | 34 | 3 (2 long, 1 short) |

A profitable `Long Exit` or `Short Exit` is reliable method-controlled evidence:
trail modes have no profit target, and the initial protective stop cannot create
a profitable stop exit. Every reference therefore covers real activation and
method-controlled exits on both sides of the market.

## Verification facts

- Every raw export has exactly two rows per sequential trade number: one Entry
  and one Exit.
- CSV closed-trade counts exactly match the metrics screenshots:
  `52, 42, 68, 47, 51, 34`.
- Profitable-trade counts, final cumulative PnL, and displayed Profit Factor
  also match their screenshots.
- Sums of the individually rounded two-decimal PnL rows can differ by a few
  cents from TradingView's higher-precision cumulative total. The screenshot
  and final cumulative field are authoritative.
- References C and E exercise the exclusive End Date close. TradingView labels
  the immediate close with the boundary bar's `07:30 UTC+8` opening timestamp
  (`23:30 UTC`), while its semantic close boundary is `08:00 UTC+8`
  (`00:00 UTC`).
- Other references close naturally before the End Date.

## Interpretation boundary

The Pine strategy uses a one-tick buffer when accepting the first method
candidate. The reviewed Merlin plan deliberately proposes a strict
protective-side comparison without that extra tick. This is a known narrow
difference, not an error in these references.

The TradingView property override disables both order-fill and realtime-tick
recalculation even though the Pine declaration defaults order-fill recalculation
to true. The accepted references therefore use close-only calculation: a
close-derived activation or trail update becomes an executable order no earlier
than the next available tick. The Phase 1 specification must preserve this
trail-update contract with synthetic long/short execution tests. This does not
require Merlin to remove its existing initial-stop protection from the entry
fill bar; that safer generic behavior remains authoritative and any exercised
difference must be documented narrowly.
