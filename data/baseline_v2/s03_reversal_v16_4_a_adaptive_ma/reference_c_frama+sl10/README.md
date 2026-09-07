# Reference C: FRAMA + Emergency SL 10%

Accepted TradingView reference using `FRAMA` length
`100`, Close Count Long/Short `6/6`,
and T Bands Long/Short `1.8/1.8%`.

Emergency SL is enabled at `10%` with an update interval of `16` bars. The export contains 3 Emergency SL exits (1 Long, 2 Short).

The raw export contains 62 closed trades (31 Long,
31 Short) and matches the TradingView metrics screenshot. The final
trade exercises the exclusive End Date close. TradingView labels that immediate
close with the boundary bar's `07:30 UTC+8` opening timestamp; its semantic close
boundary is `08:00 UTC+8` (`00:00 UTC`).
