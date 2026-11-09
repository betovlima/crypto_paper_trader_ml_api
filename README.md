# Crypto Paper Trader ML API

Independent API for long-period LightGBM walk-forward backtests. It is separate from the paper-trading API and never sends real orders.

## BTC Directional Change V12: retrospective BTC highs/lows and confirmed reversals

V12 adds a fourth VISIBLE research type: BTC maxima/minima using Directional
Change. It reuses exactly the existing three toolbar actions (Create, Research,
Export ZIP). The ML API does not run LightGBM, OOS or real brokerage orders.

Four thresholds were frozen before reviewing the results:

* 1 percent reversal from the most recent closing-price pivot.
* 2 percent reversal.
* 3 percent reversal.
* Volatility-adaptive reversal: 1.2 times past-24-hour close-return
  volatility scaled to 24 hours, capped from 1 to 4 percent. The threshold
  is frozen at the time the candidate pivot is observed, and cannot see
  prices or volatility from future candles.

A historical BOTTOM pivot is a lowest candle CLOSE. It becomes a CONFIRMED
BOTTOM only when a later CLOSE increases at least the threshold. A TOP
requires a corresponding later CLOSE decline. Signals alternate, and BTC/CASH
fills happen at the NEXT candle OPEN after confirmation, never at the
historical pivot price. Historical absolute OHLC low/high prices are
reported separately and NEVER used for trading signals.

A timestamp gap resets the event detector, suppresses an entry across the
gap, and exits an existing BTC holding at the next known candle opening.
Only one action per candle is allowed. At each partition end, any open
position is liquidated at the final known CLOSE and recorded as such.

The chronological development partition starts after the first 5 percent
of pre-OOS history and ends before 60 percent with a 48-candle embargo.
Validation is the last 40 percent strictly BEFORE the first archived OOS.
The first OOS itself stays unexamined during the automated study.
Both partitions restart at 10,000 USD equivalent with Alpaca dynamic
Taker fee tiers, full capital reinvestment and separate slippage. CASH
and BTC Buy-and-Hold are comparable baselines.

The Front displays absolute OHLC extremes and historical closing-price
pivots, plus their later confirmation points. A price chart shows recent
validation prices, pivot timestamps and confirmation timestamps separately.

The existing Export ZIP adds these files:
* directional_change.json: frozen threshold definitions and saved metrics.
* directional_change_portfolios.csv: all four strategies and time partitions.
* directional_change_extrema.csv: ALL historical pivot prices and timestamps,
  confirmation prices and timestamps, thresholds, confirmation lags.
* directional_change_trades.csv: ALL fills at next opening, entry/exit
  reason, pivot, confirmation, price, commission, slippage and compounded
  account equity.
* source_historical_audit.zip: immutable BTC candles as originally archived.

Limits: BTCUSDT represents an approximate USD-equivalent proxy to Alpaca
BTCUSD; archived 30-minute OHLCV lacks the order book, spreads, actual
Alpaca fills and intrabar ordering. Previously inspected historical OOS
does not count as a fresh independent external validation. These are
research results, not authorization for live trading.

## BTC event-driven research V11: CUSUM and Triple Barrier

V11 adds a new **"BTC: CUSUM + Triple Barrier"** mode to the existing
research mode selector. It uses exactly the same three top buttons:
**Create Strategy**, **Research BTC Events**, **Export ZIP**.
It needs only the BTC OHLCV source archive already stored inside the ML API.
There are no external scripts, no new environment parameters, no downloads
for each run, and no live brokerage operations.

This initial study deliberately **does not train LightGBM** or run OOS.
It asks whether four predetermined BTC price-event rules have any net
economic evidence before adding an ML model:

1. Buy BTC on a confirmed positive CUSUM breakout.
2. Buy BTC on a confirmed negative CUSUM move (contrarian dip).
3. Buy BTC if a confirmed negative CUSUM move is followed by a recovery
   close on the immediately following candle.
4. Buy BTC on a positive CUSUM event only in a high-volatility regime.

Both a perpetually idle **CASH** account and a **BUY_HOLD** BTC account
serve as baselines. They are charged under the same Alpaca Taker tier model
and slippage assumptions as the event policies.

### Fixed mathematical assumptions (before looking at OOS)

- Input: archived 30-minute BTC/USDT OHLCV. More granular candles can be
  supported in future studies if they are actually archived.
- CUSUM operates on confirmed close-to-close log returns with threshold
  \`max(0.8%, 2.2 * past-24h-vol * sqrt(3600/bar_seconds))\`.
- Regime filter is \`past-24h-vol > 1.25 * past-7d-vol\`.
- Study labels use a six-hour vertical barrier and adaptive price thresholds
  fixed at each candidate event:
  upper \`max(1.2%, 2 * sigma * sqrt(number_of_6h_bars))\`;
  lower \`max(0.9%, 1.3 * sigma * sqrt(number_of_6h_bars))\`.
- If high AND low cross their barriers in the **same candle**, their order
  cannot be inferred: study label is **AMBIGUOUS_BOTH_CONSERVATIVE_LOSS**.
- Trade signals are visible only after the event candle closes. All orders
  are filled at the **next opening price**, with dynamic Alpaca Taker fees
  plus separately estimated slippage. Barrier touches may prompt an exit
  only at the NEXT candle opening, never at an invented intrabar price.
- Only one buy/sell operation per candle. Open positions at segment end are
  liquidated at the final observed close and explicitly documented.
- Gaps in the archived candles are recorded, and a gap is never treated as
  a price event. Gap exits are counted separately.

### Chronological research contract

- Use **only candles strictly before the first archived OOS boundary**.
- The initial 60% of that history is development, with a 24-bar embargo;
  the remaining 40% is untouched pre-OOS validation.
- Four policy rules are frozen before running them. Any candidate suggested
  for further investigation is selected **only by development profitability
  and risk**, then checked on pre-OOS validation.
- \`RESEARCH_COMPLETE\` is a diagnostic study status, not permission to
  trade. \`result_status\` indicates whether the fixed rules offer sufficient
  *internal* evidence. **No ML strategy is created and no automatic OOS/live
  trading job is scheduled**, even if the caller sends \`auto_execute=true\`.
- Old strategies and their immutable audits keep their previous behavior.

### Audit export

The existing **Export ZIP** action includes:

- \`event_research.json\`: fixed parameters, time boundaries, labels,
  profitability and risks for all four policies and both baselines.
- \`event_research_portfolios.csv\`: development and validation finance.
- \`event_trade_actions.csv\`: COMPLETE, reproducibly reconstructed trade
  log, by policy and period, with timestamps, prices, costs and exit reasons.
- \`source_historical_audit.zip\`: original archived historical data.

**Limits:** BTC/USDT from Binance is only a USD-equivalent proxy for Alpaca
BTC/USD. The 30-minute OHLCV bars cannot reconstruct intrabar order
sequence, order-book imbalance, true Alpaca bid/ask spreads, liquidation
flow, or market execution at stop levels. A future 1-minute or tick archive
is required to investigate these more precisely. This is not a validated
trading system and not an independent test of a previously seen OOS period.

## BTC/CASH V10: stress of the *same operations* and forecast-quality diagnostics

V10 fixes an important flaw in V9: increasing execution costs could cause
the policy to fail its buy threshold, perform **zero operations** and end
with $10,000 in CASH. That was not a valid measure of the effect of costs on
the original traded strategy.

- Baseline optimization still uses the causal prior-close to next-open
  BTC/CASH engine with Alpaca taker fees and slippage.
- V10 retains the **entire original pre-OOS fill schedule**, including
  every buy, sell and the final liquidation. It then replays the same
  candle indices, prices, sides and execution times under 25% higher
  hypothetical commissions and 2x slippage.
- The replay recomputes BTC received, USD received, compounded portfolio
  value and the actual Alpaca 30-day fee tier based on newly computed fill
  notionals. It never reruns buy/exit filters, so higher cost **cannot
  erase operations**.
- The replay is checked against the original simulation using the original
  rates (fee multiplier 1, original slippage) and must reproduce the same
  capital, fees, Sharpe and drawdown. The robustness gate also rejects
  stress results without the same operation count and frozen-schedule proof.
- V9's minimum-cycle, temporal distribution, positive net profit,
  gain/loss and concentration requirements remain unchanged. Winner
  selection uses the lower of normal and *same-fill stressed* risk-adjusted
  capital scores.
- Research ZIPs and collapsed Front diagnostics include normal versus
  stressed capital and operation counts, stressed fees and slippage, and
  reasons for rejecting candidates.

### Understanding whether LightGBM predicts *magnitude*

For each 2h, 6h and 24h horizon, V10 records in its **PRE-OOS validation
partition only**:

* Return MAE and RMSE from LightGBM, compared with a constant return
  estimated from the earlier **training** partition. A positive skill
  score means fewer errors than that simple baseline; negative means worse.
* Mean predicted versus realized returns, bias, correlation and directional
  accuracy.
* Ratio of predicted-return standard deviation to the observed-return
  standard deviation. Near-zero values reveal that forecasts are too flat
  to discriminate large BTC movements.
* Five ranked prediction buckets, with realized average return for each.
  Overlapping future horizons mean observations are not independent
  samples. The data remain descriptive and are **never features or
  inputs to the live decision policy**.

The Front retains the same **three buttons at the top** (Create Strategy,
Optimize and Run, Export ZIP). Technical diagnostics stay in collapsible
sections and are available even for NO_TRADES/INSUFFICIENT_EVIDENCE.

**Research limitation:** The V8 and V9 historical OOS periods have already
been examined. While V10 continues to exclude OOS candles from all
automatic parameter selection, reusing the same period cannot provide a
fresh independent external test. Prefer future, previously unseen market
observations before claiming generalization or enabling real trading.

## BTC/CASH V9: evidence-based selection before OOS

The same **three buttons** (Create, Optimize-and-Run, Export) remain at the
top of the front end; no external scripts or new configuration files.

V9 prevents the one-trade-winner problem exposed by V8. Every new joint
2h+6h+24h research candidate now requires, **on PRE-OOS data only**:

* At least **6 completed BTC/CASH cycles**, across at least 2 of 3
  consecutive validation time segments (minimum 2 cycles per active segment).
* At least 2 validation segments with positive mark-to-market returns
  and positive overall net profit after Alpaca fees/slippage.
* Average positive cycle gain at least 75% of the absolute average losing
  cycle; the largest winning trade cannot contribute more than 70% of total
  positive cycle gains. These are evidence thresholds, not trade targets.
* Positive profit in a separate cost stress simulation with the **same
  pre-OOS candles and forecasts** but 25% greater Alpaca commissions and
  2x slippage. These increases are hypothetical robustness checks,
  **not published Alpaca tariffs**.
* Ranking candidates by the lower of baseline and stressed risk-adjusted
  compound capital scores. Fees use the same official dynamic Alpaca
  30-day volume tier algorithm on both paths.

Candidates failing a check are marked \`INSUFFICIENT_EVIDENCE\`. If no
candidate survives, there is NO champion and NO automatic final OOS job.
All failed checks, segment counts/returns, fee-stress capital, average gains
and losses, and trade concentration are retained in the strategy ZIP and
inside the collapsible research diagnostics.

For diagnostic purposes the model's weighted forecast (converted to a
6-hour equivalent return) is compared with the subsequent realized 6-hour
return **only within the pre-OOS selection partition**. These realized
outcomes never become prediction inputs, trade signals, or calibration
features.

Thresholds were fixed prior to running V9 candidates. They are deliberately
conservative and may reject all eight standard configurations. This is
preferable to labeling an isolated profitable trade a validated strategy.

**Scientific caveat:** prior versions' OOS outcomes (including V8) have
already been inspected by the researchers. Although the V9 automatic
optimizer cannot read the OOS data when selecting a candidate, using the
SAME historic OOS after this research iteration cannot establish genuinely
independent external generalization. New future samples or a newly reserved
untouched period are needed for that stronger claim.

## BTC/CASH V8: Alpaca Crypto trading fees and net reinvested profit

The **new** LightGBM BTC/CASH research campaigns model the published
Alpaca Crypto spot fee schedule by default. Existing V7 and earlier
strategies preserve their original fixed-fee economic results.

- Eight official 30-day USD notional volume tiers: maker 0.15%-0%,
  taker 0.25%-0.10%; both BUY and SELL volume count.
- Default execution is **TAKER**, because a next-candle market fill cannot
  claim maker liquidity. A maker quote is implemented for explicit future
  configuration, but never assumed in the default backtest.
- Daily fee tier snapshot at 03:00 New York, computed from completed
  simulated fills in the previous 30 days only. External account trading
  activity is unknown and assumed zero; this may overstate actual fees.
- BUY: Alpaca fee is deducted from **BTC credited**; SELL: deducted
  from **USD credited**. Both are also accounted for as USD-equivalent
  fees in the audit.
- Slippage/spread is a **separate estimate**, not misrepresented as an
  additional Alpaca commission. Initial capital, every reinvested trade,
  fees, slippage, capital curve, drawdown and final net USD-equivalent
  profit are included in the simulation.
- The untouched OOS benchmark **Buy and Hold** is charged using the
  same Alpaca fee schedule for fairness.
- A separate Alpaca Crypto account-fee reconciliation should eventually
  query actual account activities \`CFEE\` / \`FEE\`. This research has no
  authenticated Alpaca brokerage access and cannot know the user's
  true commission tier or any fees imposed by intermediaries.

Authoritative documentation:
https://docs.alpaca.markets/us/docs/crypto-fees

**Important:** archived candles are BTC/USDT from MEXC/Binance, while a
future Alpaca trade uses BTC/USD. Present results are USD-equivalent
hypothetical simulations, not actual Alpaca execution prices or realized
taxable income. Neither exchange spreads nor third-party charges can
be reconstructed exactly from source candles. The simulator does NOT
send orders. Each Alpaca crypto order may be subject to a $200,000
notional order limit and eligibility requirements.

### Net capital formula

- BUY at next open: BTC credited = (cash / (open * (1 + slippage)))
  * (1 - maker/taker fee).
- SELL at next open: cash credited = BTC * open * (1 - slippage)
  * (1 - maker/taker fee).
- Net strategy profit = final cash - initial cash; all fees and estimated
  slippage have already been deducted. The report separately records
  Alpaca fees, slippage estimates, tier at each fill, and fee asset units.

## BTC/CASH V7: probabilistic entry thresholds and broader pre-OOS validation

The V6 research workflow still uses **only three top-of-screen actions**.
V7 fixes two simultaneous entry blockers: the minimum probability has
become a searched policy threshold (including probabilities below 50%), and
a horizon with a positive predicted return now contributes a positive vote
regardless of its probability being below 50%. The calibrated aggregate
probability remains a separate decision filter, as does expected return
after fees/slippage.

The probability calibrator evaluates identity, Platt on raw probabilities
and Platt on log-odds using a **chronological tail of each horizon's own
calibration partition**. Selection/test labels are not used for calibration
method choice. Audit records the method, calibration Brier and sample count.

The economic selection window is now the last 25% of up to 48,000 available
candles preceding the **first untouched OOS boundary**, compared with 13%
of up to 16,000 candles in V6. One reinvested BTC/CASH portfolio is
evaluated with three chronological risk segments. The final OOS is never
used to choose hyperparameters, policy thresholds or calibration methods.

The goal is compound net capital growth with risk controls, not the number
of trades. These changes allow opportunities to be evaluated, but do not
guarantee profitable trades.

## BTC/CASH V5: compound capital and causal rotation research

The **Laboratório ML** page is the only user-facing workflow. LightGBM
provides forecast probabilities and returns for 2h, 6h and 24h; the research
objective is now **one fully reinvested BTC/CASH portfolio**, not accuracy, AUC,
or the average of three independent simulated accounts.

Creating a strategy selects an audited historical run. Either **Otimizar
rotações BTC/CASH** (8 Latin Hypercube candidates) or **CARO · Otimizar
capitalização** (24 adaptive GP candidates) searches model hyperparameters
**and decision-policy parameters**:

- minimum upside probability and net predicted return required to enter;
- weaker forecast conditions to exit (probability hysteresis);
- minimum number of favorable horizons; forecast-horizon weights;
- close-confirmed stop loss, trailing stop and take-profit thresholds;
- minimum holding candles and cooldown candles before re-entry.

Every simulated buy reinvests the full current cash balance. Every sale
returns the net BTC value to CASH. Historical fees and slippage are applied
to both sides; no order is ever sent to an exchange or broker.

Training, early stopping, probability calibration and economic selection
happen strictly before the first immutable OOS fold. Signals are derived
from the previous candle CLOSE, with orders only at the next candle OPEN.
Stops are close-confirmed and also operate if a new forecast is missing;
we never claim an intrabar fill at an unobserved stop price.

The selected model and decision-policy snapshot are stored with the new
backtest job. The final OOS capitalization uses the identical policy engine
with **no re-optimization on OOS data**. For optimized strategy jobs the
Front automatically displays the BTC/CASH equity curve, executed actions,
drawdown and Buy & Hold benchmark, without a model-versus-model comparison.

**Audit ZIP:** choose Exportar relatório e auditoria ZIP on the selected
strategy after its final run completes. The package contains:
`strategy.json`, `optimization_trials.csv`, `source_historical_audit.zip`,
`final_execution_parameters.json`, `final_execution_result.json`,
`final_oos_audit.zip` and `final_oos_capitalization.json`.
The last file includes the **complete** chronological equity timeline,
all simulated transactions and fees, the chosen risk policy and OOS returns.

A successful pre-OOS selection does **not** establish live profitability.
The V5 search space is exploratory and does not yet import the 52 engineered
features from the MCT. Historical snapshots remain immutable.

## LightGBM replacement (V3 research)

The classification and regression learners now use LightGBM 4.6.x, including
both BTC Direction & Return and the legacy target/stop backtest. Old jobs and
audit ZIPs are **not rewritten or relabelled**. New backtests have immutable
LightGBM parameter snapshots; old XGBoost-specific keys stay in the database
for history but are hidden in the parameters UI.

The LightGBM section exposes num_leaves, min_child_samples, max_bin,
sampling frequency, regularization and boosting settings. Regression has
independent num_leaves and min_child_samples. New forecasts must be trained
again: swapping an existing replay ZIP does not convert its saved predictions.

## Version 0.2.2 — database-backed Railway CORS and explicit-window pagination

### MEXC history collection fix

Historical candles are now fetched through explicit, non-overlapping `startTime`/`endTime` windows. This avoids the real MEXC behavior where an `endTime`-only request can repeatedly return the same recent page and leave only 499 closed candles after the current candle is removed. The window size is stored in the runtime database as `market.page_window_candles`.

Every operational parameter is persisted in `ml_runtime_parameters` and loaded from the database at runtime:

- CORS, worker concurrency and API listing limits;
- frontend progress polling and recent-execution limits;
- MEXC URL, timeout, pagination, warm-up and candle limits;
- symbols, periods, accepted timeframes and form defaults;
- capital, fees, slippage and probability thresholds;
- feature-engineering periods and Fibonacci levels;
- entry and exit label rules;
- LightGBM hyperparameters and calibration;
- walk-forward windows and fold limits;
- stops, targets, trailing rules and reporting limits.

The database row stores the value, type, constraints, options, category, description, visibility and restart requirement. Existing rows are not overwritten during startup. The database is the operational source of truth.

A small installation database, `ml_parameters_seed.db`, contains the initial catalog for a brand-new runtime database. Parameter values are not declared in Python, `.env`, JSON, YAML or JavaScript files. Existing runtime rows are never overwritten by the installation database.

New jobs receive an immutable complete snapshot in:

```text
ml_backtest_jobs.settings_snapshot_json
```

Changing a parameter affects only future jobs. Running and completed jobs preserve the exact configuration with which they started.

No market, model, indicator, threshold, timeframe, fee, risk or simulation parameter is read from `.env` or another runtime configuration file.

`DATABASE_URL` is the only bootstrap value because the service must locate its database before database parameters can be loaded.

## Local execution

```bash
poetry install
poetry run uvicorn crypto_paper_trader_ml_api.app:app --app-dir src --host 0.0.0.0 --port 8001 --reload
```

Open:

```text
http://localhost:8001/docs
```

## Configuration endpoints

```http
GET /health
GET /v1/ml-configuration
GET /v1/ml-settings
PUT /v1/ml-settings
```

- `/v1/ml-configuration` supplies database defaults, limits, timeframe options and frontend behavior.
- `/v1/ml-settings` returns the complete database catalog.
- `PUT /v1/ml-settings` validates and persists changed rows.

Some parameters, such as CORS and worker concurrency, are marked `requires_restart` because those components are initialized when the process starts.

## Replay V1 × V2 pelo Front

Na aba BTC Forecast, selecione a execução original concluída do BTC
Direction & Return com auditoria armazenada no banco. Clique em
**Reexecutar e comparar V1 × V2**. A ML API cria uma nova execução
que usa exclusivamente os candles persistidos, valida os folds e
apresenta as métricas comparadas no próprio Front.

A integração utiliza POST /v1/ml-backtests/{id}/replay e registra
replay_parent_job_id na execução nova. O resultado completo fica
em replay_comparison. Não há scripts ou caminhos de ZIP no fluxo
do usuário.

## Backtest endpoints

```http
POST   /v1/ml-backtests
GET    /v1/ml-backtests
GET    /v1/ml-backtests/{id}
GET    /v1/ml-backtests/{id}/progress
GET    /v1/ml-backtests/{id}/results
GET    /v1/ml-backtests/{id}/trades
DELETE /v1/ml-backtests/{id}
```

The service downloads closed public MEXC Spot candles, calculates technical features, creates entry and exit labels, performs expanding walk-forward validation, calibrates probabilities and simulates out-of-sample operations with costs.

The history collector continues paging after a 499-candle response. It requests older data until the database target is satisfied or MEXC has no earlier history.

## Railway

Create a separate Railway service and attach a persistent volume at `/data`.

The only ML API bootstrap variable is:

```env
DATABASE_URL=sqlite:////data/ml_backtests.db
```

After the first startup, configure `api.cors_origins` in the frontend parameter screen or through `PUT /v1/ml-settings`, then restart the ML API.

The frontend still needs the two independent service addresses:

```env
VITE_API_URL=https://YOUR-PAPER-API.up.railway.app
VITE_ML_API_URL=https://YOUR-ML-API.up.railway.app
```

These values are network addresses required to locate the services. They are not trading or Machine Learning parameters.

### CORS in Railway

Version 0.2.2 resolves CORS origins dynamically from `api.cors_origins` in the runtime database. The production frontend origin is included in the seed catalog and merged into existing runtime databases during startup. Changing the parameter no longer requires a service restart.
