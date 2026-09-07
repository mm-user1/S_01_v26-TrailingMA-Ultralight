# Reference D: DSMA + Emergency SL 10%

Accepted TradingView reference using `DSMA` length
`200`, Close Count Long/Short `6/7`,
and T Bands Long/Short `1.6/1.8%`.

Emergency SL is enabled at `10%` with an update interval of `16` bars. The export contains 3 Emergency SL exits (0 Long, 3 Short).

The raw export contains 53 closed trades (27 Long,
26 Short) and matches the TradingView metrics screenshot. The final
trade exercises the exclusive End Date close. TradingView labels that immediate
close with the boundary bar's `07:30 UTC+8` opening timestamp; its semantic close
boundary is `08:00 UTC+8` (`00:00 UTC`).
