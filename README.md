# Crypto Paper Trader ML API

Independent API for long-period LightGBM walk-forward backtests. It is separate from the paper-trading API and never sends real orders.

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
