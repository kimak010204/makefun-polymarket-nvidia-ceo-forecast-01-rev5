# Polymarket NVIDIA CEO forecast — revision 5

Forecasts the 5 quantiles of the next-CEO-of-NVIDIA Polymarket prices
4 hours ahead, at 12 checkpoints, from minute-by-minute Yes prices.

Revision 5 keeps revision 4's median exactly (last price + sum-reversion
drift + asymmetric up-spike give-back, regime gate) and replaces the five
scalar width scales with per-horizon-band scales (h<60, 60-120, 120-180,
180-240 min), fitted by coordinate descent on the official pinball loss
over the practice (6 cases) + fresh (4 cases) splits. The width-vs-board
deviation slope rises 1.5 -> 2.25 and the empirical-vs-baseline blend
shifts 0.85 -> 0.80.

Validated (official pinball loss, same protocol as revisions 2-4):

| split | revision 4 | this program |
|---|---|---|
| practice (6 cases) | 0.257016 -> 0.255974 | **0.255155** |
| fresh Nov 12-13 (4 cases) | 0.205365 -> 0.205597 | **0.203421** |
| chaotic Nov 10-11 (11 cases, hold-out) | 2.666946 -> 2.665082 | **2.660023** |

Chaotic was not in the fitting objective and still improved — the main
guard against overfitting the 10 fitting cases. Median error unchanged
(0.7144); this revision does not close the median gap to the leaders.

## Run

```sh
python3 main.py   # reads /input, writes /output (grader)
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Standard library only, no network, deterministic.
