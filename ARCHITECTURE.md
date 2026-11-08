# Architecture

## Independent services

```text
Frontend
├── Paper Trading API
└── Machine Learning Backtest API
```

The ML API owns its own database, parameter catalog, LightGBM jobs and results. It does not change the live paper-trading API.

## Parameter flow

```text
ml_runtime_parameters
        ├── values
        ├── types and constraints
        ├── options
        ├── descriptions
        └── restart requirements
                 ↓
GET /v1/ml-configuration and GET /v1/ml-settings
                 ↓
Frontend forms and parameter editor
                 ↓
PUT /v1/ml-settings
                 ↓
POST /v1/ml-backtests
                 ↓
Resolved request + immutable database snapshot
                 ↓
ml_backtest_jobs.settings_snapshot_json
                 ↓
Market data → features → labels → LightGBM → walk-forward → simulation
```

Every job uses the snapshot captured at creation. Editing the current catalog does not mutate running or historical executions.

## Database source of truth

At first installation, the service copies missing rows from the packaged `ml_parameters_seed.db` into the runtime database. Existing rows, including their values and metadata, are not overwritten on later startups. Parameter values are stored in databases rather than source or configuration files.

Runtime validation reads type, minimum, maximum and options from the database rows. Cross-parameter validation protects relationships such as:

- default period inside minimum and maximum limits;
- default timeframes inside the enabled timeframe lists;
- frontend list size inside the API limit;
- probability defaults inside the accepted range;
- ordered EMA and Fibonacci periods;
- valid walk-forward fold and training windows.

## Bootstrap boundary

`DATABASE_URL` is the only external bootstrap value. It cannot be stored in the same database that it is required to locate.

The browser also requires `VITE_API_URL` and `VITE_ML_API_URL` to locate the two HTTP services. These are infrastructure addresses, not analytical parameters.

## Main modules

- `database.py`: jobs, parameter table, SQLite initialization and schema compatibility.
- `seed/ml_parameters_seed.db`: database containing the initial catalog for a new installation.
- `settings_service.py`: database loading, row-based validation, updates and immutable snapshots.
- `mexc.py`: database-parameterized paginated closed-candle collection.
- `features.py`: database-parameterized features and labels.
- `engine.py`: LightGBM, calibration, walk-forward simulation and metrics.
- `service.py`: asynchronous jobs and snapshot enforcement.
- `app.py`: REST endpoints.


## Historical market-data pagination

The MEXC history adapter traverses the requested period forward through explicit, non-overlapping `startTime` and `endTime` windows. Both the request limit and the number of candle intervals per window are database-backed parameters. Empty windows are allowed so pairs listed after the requested start date can still be processed.
