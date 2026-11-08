from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from .database import MLRuntimeParameter, SessionLocal


@dataclass(frozen=True)
class ParameterDefinition:
    key: str
    category: str
    label: str
    description: str
    value_type: str
    default: Any
    minimum: float | int | None = None
    maximum: float | int | None = None
    options: tuple[Any, ...] = ()
    requires_restart: bool = False
    public: bool = True
    order: int = 0


SEED_DATABASE_PATH = (
    Path(__file__).resolve().parent / "seed" / "ml_parameters_seed.db"
)


@lru_cache(maxsize=1)
def _seed_parameter_rows() -> tuple[dict[str, Any], ...]:
    if not SEED_DATABASE_PATH.exists():
        raise RuntimeError(
            f"ML parameter seed database was not found: {SEED_DATABASE_PATH}"
        )
    connection = sqlite3.connect(SEED_DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT key, category, label, description, value_type, value_json,
                   constraints_json, public, requires_restart, display_order
            FROM ml_runtime_parameters
            ORDER BY display_order, key
            """
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        raise RuntimeError("The ML parameter seed database contains no parameters")
    return tuple(dict(row) for row in rows)


class ParameterValidationError(ValueError):
    pass


@dataclass(frozen=True)
class RuntimeParameters:
    values: dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self.values:
            raise KeyError(f"Required runtime parameter is missing from the database: {key}")
        return self.values[key]

    def integer(self, key: str) -> int:
        return int(self.require(key))

    def floating(self, key: str) -> float:
        return float(self.require(key))

    def boolean(self, key: str) -> bool:
        return bool(self.require(key))

    def string(self, key: str) -> str:
        return str(self.require(key))

    def list(self, key: str) -> list[Any]:
        value = self.require(key)
        if not isinstance(value, list):
            raise TypeError(f"Runtime parameter {key} is not a list")
        return list(value)

    def snapshot(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.values))


def synchronize_parameter_catalog() -> None:
    """Copy missing parameter rows from the installation database.

    Existing rows are never overwritten. Values, constraints, options and
    metadata in the runtime database remain the operational source of truth.
    """
    now = datetime.now(timezone.utc)
    seed_rows = _seed_parameter_rows()
    with SessionLocal.begin() as session:
        existing_keys = {
            item.key for item in session.query(MLRuntimeParameter.key).all()
        }
        # Database migration: add the explicitly selectable research source
        # without modifying existing parameter values or storing it in .env.
        if "backtest.default_history_source" not in existing_keys:
            session.add(MLRuntimeParameter(
                key="backtest.default_history_source",
                category="Backtest",
                label="Default historical market source",
                description="MEXC historical candles or separately identified Binance Spot archives. Never blend sources.",
                value_type="string",
                value_json=json.dumps("MEXC"),
                constraints_json=json.dumps({
                    "options": ["MEXC", "BINANCE_SPOT_ARCHIVE"]
                }),
                public=True,
                requires_restart=False,
                display_order=110,
                updated_at=now,
            ))
        # Add direct forecast settings without replacing existing runtime choices.
        forecast_defaults = [
            {
                "key": "backtest.default_experiment_type",
                "category": "Backtest",
                "label": "Default experiment type",
                "description": "Target/stop strategy or independent BTC movement research.",
                "value_type": "string",
                "value": "TARGET_STOP",
                "constraints": {"options": ["TARGET_STOP", "BTC_DIRECTION_RETURN"]},
                "public": True,
                "requires_restart": False,
                "display_order": 111,
            },
            {
                "key": "forecast.horizons_hours",
                "category": "BTC Forecast",
                "label": "Forecast horizons (hours)",
                "description": "Chronologically validated future BTC return horizons.",
                "value_type": "integer_list",
                "value": [2, 6, 24],
                "constraints": {"minimum": 1, "maximum": 72},
                "public": True,
                "requires_restart": False,
                "display_order": 112,
            },
            {
                "key": "forecast.maximum_folds",
                "category": "BTC Forecast",
                "label": "Maximum independent temporal folds",
                "description": "Maximum number of OOS folds for direct BTC forecasting.",
                "value_type": "integer",
                "value": 3,
                "constraints": {"minimum": 2, "maximum": 6},
                "public": True,
                "requires_restart": False,
                "display_order": 113,
            },
        ]
        # Regression parameters are independent from classification. Numerical
        # scales are expressed in the regressor's scaled target units.
        regression_defaults = [
          {
            "key": "forecast.regression_target_scale",
            "category": "BTC Forecast",
            "label": "Regressor target scale (basis points)",
            "description": "Editable forecast research parameter stored in the runtime database.",
            "value_type": "float",
            "value": 10000,
            "constraints": {
              "minimum": 1,
              "maximum": 100000
            },
            "public": True,
            "requires_restart": False,
            "display_order": 114
          },
          {
            "key": "forecast.regression_n_estimators",
            "category": "BTC Forecast",
            "label": "Regressor maximum XGBoost trees",
            "description": "Editable forecast research parameter stored in the runtime database.",
            "value_type": "integer",
            "value": 350,
            "constraints": {
              "minimum": 20,
              "maximum": 2000
            },
            "public": True,
            "requires_restart": False,
            "display_order": 115
          },
          {
            "key": "forecast.regression_max_depth",
            "category": "BTC Forecast",
            "label": "Regressor maximum tree depth",
            "description": "Editable forecast research parameter stored in the runtime database.",
            "value_type": "integer",
            "value": 4,
            "constraints": {
              "minimum": 1,
              "maximum": 12
            },
            "public": True,
            "requires_restart": False,
            "display_order": 116
          },
          {
            "key": "forecast.regression_learning_rate",
            "category": "BTC Forecast",
            "label": "Regressor learning rate",
            "description": "Editable forecast research parameter stored in the runtime database.",
            "value_type": "float",
            "value": 0.04,
            "constraints": {
              "minimum": 0.001,
              "maximum": 0.5
            },
            "public": True,
            "requires_restart": False,
            "display_order": 117
          },
          {
            "key": "forecast.regression_min_child_weight",
            "category": "BTC Forecast",
            "label": "Regressor minimum child weight",
            "description": "Editable forecast research parameter stored in the runtime database.",
            "value_type": "float",
            "value": 4,
            "constraints": {
              "minimum": 0,
              "maximum": 200
            },
            "public": True,
            "requires_restart": False,
            "display_order": 118
          },
          {
            "key": "forecast.regression_gamma",
            "category": "BTC Forecast",
            "label": "Regressor split penalty (in scaled target units)",
            "description": "Editable forecast research parameter stored in the runtime database.",
            "value_type": "float",
            "value": 0,
            "constraints": {
              "minimum": 0,
              "maximum": 1000
            },
            "public": True,
            "requires_restart": False,
            "display_order": 119
          },
          {
            "key": "forecast.regression_reg_alpha",
            "category": "BTC Forecast",
            "label": "Regressor L1 regularization (in scaled target units)",
            "description": "Editable forecast research parameter stored in the runtime database.",
            "value_type": "float",
            "value": 0,
            "constraints": {
              "minimum": 0,
              "maximum": 1000
            },
            "public": True,
            "requires_restart": False,
            "display_order": 120
          },
          {
            "key": "forecast.regression_reg_lambda",
            "category": "BTC Forecast",
            "label": "Regressor L2 regularization",
            "description": "Editable forecast research parameter stored in the runtime database.",
            "value_type": "float",
            "value": 1.5,
            "constraints": {
              "minimum": 0,
              "maximum": 1000
            },
            "public": True,
            "requires_restart": False,
            "display_order": 121
          },
          {
            "key": "forecast.regression_early_stopping_rounds",
            "category": "BTC Forecast",
            "label": "Regressor early stopping patience",
            "description": "Editable forecast research parameter stored in the runtime database.",
            "value_type": "integer",
            "value": 30,
            "constraints": {
              "minimum": 2,
              "maximum": 300
            },
            "public": True,
            "requires_restart": False,
            "display_order": 122
          },
          {
            "key": "forecast.bootstrap_samples",
            "category": "BTC Forecast",
            "label": "Temporal bootstrap samples",
            "description": "Editable forecast research parameter stored in the runtime database.",
            "value_type": "integer",
            "value": 200,
            "constraints": {
              "minimum": 50,
              "maximum": 1000
            },
            "public": True,
            "requires_restart": False,
            "display_order": 123
          }
        ]
        forecast_defaults.extend(regression_defaults)
        for item in forecast_defaults:
            if item["key"] not in existing_keys:
                session.add(MLRuntimeParameter(
                    key=item["key"],
                    category=item["category"],
                    label=item["label"],
                    description=item["description"],
                    value_type=item["value_type"],
                    value_json=json.dumps(item["value"]),
                    constraints_json=json.dumps(item["constraints"]),
                    public=item["public"],
                    requires_restart=item["requires_restart"],
                    display_order=item["display_order"],
                    updated_at=now,
                ))
        for seed in seed_rows:
            if seed["key"] in existing_keys:
                if seed["key"] == "api.cors_origins":
                    existing = session.get(MLRuntimeParameter, seed["key"])
                    if existing is not None:
                        current_values = json.loads(existing.value_json)
                        seed_values = json.loads(seed["value_json"])
                        merged_values = list(dict.fromkeys([*current_values, *seed_values]))
                        if merged_values != current_values:
                            existing.value_json = json.dumps(merged_values)
                            existing.updated_at = now
                        existing.requires_restart = False
                        existing.description = seed["description"]
                continue
            session.add(
                MLRuntimeParameter(
                    key=seed["key"],
                    category=seed["category"],
                    label=seed["label"],
                    description=seed["description"],
                    value_type=seed["value_type"],
                    value_json=seed["value_json"],
                    constraints_json=seed["constraints_json"],
                    public=bool(seed["public"]),
                    requires_restart=bool(seed["requires_restart"]),
                    display_order=int(seed["display_order"]),
                    updated_at=now,
                )
            )


def get_runtime_parameters() -> RuntimeParameters:
    with SessionLocal() as session:
        rows = session.query(MLRuntimeParameter).all()
    values = {row.key: json.loads(row.value_json) for row in rows}
    required_keys = {row["key"] for row in _seed_parameter_rows()}
    if not required_keys.issubset(values):
        synchronize_parameter_catalog()
        with SessionLocal() as session:
            rows = session.query(MLRuntimeParameter).all()
        values = {row.key: json.loads(row.value_json) for row in rows}
    missing = sorted(required_keys.difference(values))
    if missing:
        raise RuntimeError(
            "Required ML parameters are missing from the database: "
            + ", ".join(missing)
        )
    return RuntimeParameters(values)


def get_runtime_parameters_from_snapshot(snapshot: dict[str, Any]) -> RuntimeParameters:
    current = get_runtime_parameters().snapshot()
    missing = set(current).difference(snapshot)
    if missing:
        current.update(snapshot)
        snapshot = current
    return RuntimeParameters(snapshot)


def _validate_scalar(definition: ParameterDefinition, value: Any) -> Any:
    value_type = definition.value_type
    if value_type == "string":
        if not isinstance(value, str) or not value.strip():
            raise ParameterValidationError(f"{definition.key} must be a non-empty string")
        normalized: Any = value.strip()
    elif value_type == "integer":
        if isinstance(value, bool):
            raise ParameterValidationError(f"{definition.key} must be an integer")
        try:
            normalized = int(value)
        except (TypeError, ValueError) as exc:
            raise ParameterValidationError(f"{definition.key} must be an integer") from exc
    elif value_type == "float":
        if isinstance(value, bool):
            raise ParameterValidationError(f"{definition.key} must be numeric")
        try:
            normalized = float(value)
        except (TypeError, ValueError) as exc:
            raise ParameterValidationError(f"{definition.key} must be numeric") from exc
    elif value_type == "boolean":
        if not isinstance(value, bool):
            raise ParameterValidationError(f"{definition.key} must be boolean")
        normalized = value
    else:
        raise ParameterValidationError(f"Unsupported scalar type for {definition.key}: {value_type}")

    if definition.minimum is not None and normalized < definition.minimum:
        raise ParameterValidationError(f"{definition.key} must be >= {definition.minimum}")
    if definition.maximum is not None and normalized > definition.maximum:
        raise ParameterValidationError(f"{definition.key} must be <= {definition.maximum}")
    if definition.options and normalized not in definition.options:
        raise ParameterValidationError(
            f"{definition.key} must be one of: {', '.join(map(str, definition.options))}"
        )
    return normalized


def validate_parameter_value(definition: ParameterDefinition, value: Any) -> Any:
    if definition.value_type in {"string", "integer", "float", "boolean"}:
        return _validate_scalar(definition, value)

    if not isinstance(value, list) or not value:
        raise ParameterValidationError(f"{definition.key} must be a non-empty list")
    item_type = definition.value_type.removesuffix("_list")
    item_definition = ParameterDefinition(
        key=definition.key,
        category=definition.category,
        label=definition.label,
        description=definition.description,
        value_type=item_type,
        default=None,
        minimum=definition.minimum,
        maximum=definition.maximum,
        options=definition.options,
    )
    normalized = [_validate_scalar(item_definition, item) for item in value]
    if len(set(map(str, normalized))) != len(normalized):
        raise ParameterValidationError(f"{definition.key} cannot contain duplicate values")
    return normalized



def _validate_cross_parameter_values(values: dict[str, Any]) -> None:
    default_recent_limit = int(values["api.recent_jobs_default_limit"])
    maximum_recent_limit = int(values["api.recent_jobs_maximum_limit"])
    frontend_recent_limit = int(values["frontend.recent_jobs_limit"])
    if default_recent_limit > maximum_recent_limit:
        raise ParameterValidationError("The default recent-job limit exceeds the API maximum")
    if frontend_recent_limit > maximum_recent_limit:
        raise ParameterValidationError("The frontend recent-job limit exceeds the API maximum")

    symbol_minimum = int(values["backtest.symbol_min_length"])
    symbol_maximum = int(values["backtest.symbol_max_length"])
    if symbol_minimum > symbol_maximum:
        raise ParameterValidationError("The minimum symbol length exceeds the maximum")

    decision_timeframes = [str(value) for value in values["backtest.allowed_decision_timeframes"]]
    trend_timeframes = [str(value) for value in values["backtest.allowed_trend_timeframes"]]
    if not decision_timeframes or not trend_timeframes:
        raise ParameterValidationError("The database timeframe option lists cannot be empty")
    if str(values["backtest.default_decision_timeframe"]) not in decision_timeframes:
        raise ParameterValidationError("The default decision timeframe is not enabled")
    if str(values["backtest.default_trend_timeframe"]) not in trend_timeframes:
        raise ParameterValidationError("The default trend timeframe is not enabled")

    minimum_years = int(values["backtest.minimum_period_years"])
    maximum_years = int(values["market.max_backtest_years"])
    default_years = int(values["backtest.default_period_years"])
    if minimum_years > maximum_years or not minimum_years <= default_years <= maximum_years:
        raise ParameterValidationError("The default backtest period must be inside the database period limits")

    if float(values["backtest.default_initial_capital"]) > float(values["backtest.maximum_initial_capital"]):
        raise ParameterValidationError("The default initial capital exceeds the database maximum")
    if float(values["backtest.default_fee_rate"]) > float(values["backtest.maximum_fee_rate"]):
        raise ParameterValidationError("The default fee rate exceeds the database maximum")
    if float(values["backtest.default_slippage_rate"]) > float(values["backtest.maximum_slippage_rate"]):
        raise ParameterValidationError("The default slippage rate exceeds the database maximum")

    threshold_minimum = float(values["backtest.minimum_probability_threshold"])
    threshold_maximum = float(values["backtest.maximum_probability_threshold"])
    if threshold_minimum > threshold_maximum:
        raise ParameterValidationError("The minimum probability threshold exceeds the maximum")
    for key in ("backtest.default_entry_threshold", "backtest.default_exit_threshold"):
        if not threshold_minimum <= float(values[key]) <= threshold_maximum:
            raise ParameterValidationError(f"{key} must be inside the database threshold limits")

    ema_periods = [int(value) for value in values["features.ema_periods"]]
    if len(ema_periods) < 4 or any(value <= 0 for value in ema_periods):
        raise ParameterValidationError("features.ema_periods must contain at least four positive periods")
    if ema_periods != sorted(ema_periods):
        raise ParameterValidationError("features.ema_periods must be sorted from fastest to slowest")

    return_periods = [int(value) for value in values["features.return_periods"]]
    if any(value <= 0 for value in return_periods):
        raise ParameterValidationError("features.return_periods must contain positive periods")
    fibonacci_levels = [float(value) for value in values["features.fibonacci_levels"]]
    if fibonacci_levels != sorted(fibonacci_levels) or any(not 0 < value < 1 for value in fibonacci_levels):
        raise ParameterValidationError("features.fibonacci_levels must be sorted values between zero and one")
    if int(values["features.trend_ema_fast"]) >= int(values["features.trend_ema_slow"]):
        raise ParameterValidationError("The fast trend EMA must be lower than the slow trend EMA")

    if int(values["walk_forward.minimum_folds"]) > int(values["walk_forward.maximum_folds"]):
        raise ParameterValidationError("The minimum walk-forward folds exceed the maximum")
    if int(values["walk_forward.initial_train_minimum"]) >= int(values["walk_forward.minimum_usable_candles"]):
        raise ParameterValidationError("The initial training minimum must be lower than the minimum usable candles")

    if values.get("backtest.default_experiment_type", "TARGET_STOP") not in {
        "TARGET_STOP", "BTC_DIRECTION_RETURN",
    }:
        raise ParameterValidationError("Unsupported experiment type")
    horizons = [int(value) for value in values.get("forecast.horizons_hours", [2, 6, 24])]
    if not horizons or len(set(horizons)) != len(horizons) or any(h <= 0 or h > 72 for h in horizons):
        raise ParameterValidationError("Forecast horizons must be unique and between 1 and 72 hours")
    if not 2 <= int(values.get("forecast.maximum_folds", 3)) <= 6:
        raise ParameterValidationError("Forecast maximum folds must be between 2 and 6")

    scale = float(values.get("forecast.regression_target_scale", 10000.0))
    if not 1 <= scale <= 100000:
        raise ParameterValidationError("Regressor target scale must be between 1 and 100000")
    samples = int(values.get("forecast.bootstrap_samples", 200))
    if not 50 <= samples <= 1000:
        raise ParameterValidationError("Bootstrap samples must be between 50 and 1000")
    regression_trees = int(values.get("forecast.regression_n_estimators", 350))
    patience = int(values.get("forecast.regression_early_stopping_rounds", 30))
    if patience >= regression_trees:
        raise ParameterValidationError("Regressor early stopping must be shorter than max trees")

    if values.get("backtest.default_history_source", "MEXC") not in {
        "MEXC", "BINANCE_SPOT_ARCHIVE",
    }:
        raise ParameterValidationError("Unsupported default history source")

    if int(values["market.max_candles"]) < int(values["market.minimum_raw_candles"]):
        raise ParameterValidationError("market.max_candles must be greater than the minimum raw candles")
    if int(values["market.page_window_candles"]) > int(values["market.page_limit"]):
        raise ParameterValidationError(
            "market.page_window_candles cannot exceed market.page_limit"
        )


def list_parameter_rows(*, public_only: bool = False) -> list[dict[str, Any]]:
    with SessionLocal() as session:
        query = session.query(MLRuntimeParameter)
        if public_only:
            query = query.filter(MLRuntimeParameter.public.is_(True))
        rows = query.order_by(MLRuntimeParameter.display_order, MLRuntimeParameter.key).all()
        return [
            {
                "key": row.key,
                "category": row.category,
                "label": row.label,
                "description": row.description,
                "value_type": row.value_type,
                "value": json.loads(row.value_json),
                "constraints": json.loads(row.constraints_json or "{}"),
                "public": row.public,
                "requires_restart": row.requires_restart,
                "updated_at": row.updated_at,
            }
            for row in rows
        ]


def _definition_from_database_row(row: MLRuntimeParameter) -> ParameterDefinition:
    constraints = json.loads(row.constraints_json or "{}")
    return ParameterDefinition(
        key=row.key,
        category=row.category,
        label=row.label,
        description=row.description,
        value_type=row.value_type,
        default=json.loads(row.value_json),
        minimum=constraints.get("minimum"),
        maximum=constraints.get("maximum"),
        options=tuple(constraints.get("options") or ()),
        requires_restart=row.requires_restart,
        public=row.public,
        order=row.display_order,
    )


def update_parameter_rows(updates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    requested = list(updates)
    keys = [str(update.get("key", "")) for update in requested]
    with SessionLocal() as session:
        rows = {
            row.key: row
            for row in session.query(MLRuntimeParameter).filter(
                MLRuntimeParameter.key.in_(keys)
            ).all()
        }

    validated: list[tuple[str, Any]] = []
    for update, key in zip(requested, keys, strict=True):
        row = rows.get(key)
        if row is None:
            raise ParameterValidationError(f"Unknown runtime parameter: {key}")
        definition = _definition_from_database_row(row)
        value = validate_parameter_value(definition, update.get("value"))
        validated.append((key, value))

    current = get_runtime_parameters().snapshot()
    current.update(dict(validated))
    _validate_cross_parameter_values(current)

    now = datetime.now(timezone.utc)
    with SessionLocal.begin() as session:
        for key, value in validated:
            row = session.get(MLRuntimeParameter, key)
            if row is None:
                raise ParameterValidationError(f"Runtime parameter is missing from the database: {key}")
            row.value_json = json.dumps(value, separators=(",", ":"))
            row.updated_at = now
    return list_parameter_rows()


def public_configuration_payload() -> dict[str, Any]:
    runtime = get_runtime_parameters()
    return {
        "source": "DATABASE",
        "defaults": {
            "symbol": runtime.string("backtest.default_symbol"),
            "experiment_type": runtime.string("backtest.default_experiment_type"),
            "history_source": runtime.string("backtest.default_history_source"),
            "period_years": runtime.integer("backtest.default_period_years"),
            "decision_timeframe": runtime.string("backtest.default_decision_timeframe"),
            "trend_timeframe": runtime.string("backtest.default_trend_timeframe"),
            "initial_capital": runtime.floating("backtest.default_initial_capital"),
            "fee_rate": runtime.floating("backtest.default_fee_rate"),
            "slippage_rate": runtime.floating("backtest.default_slippage_rate"),
            "entry_threshold": runtime.floating("backtest.default_entry_threshold"),
            "exit_threshold": runtime.floating("backtest.default_exit_threshold"),
            "reinvestment": runtime.boolean("backtest.default_reinvestment"),
        },
        "limits": {
            "minimum_period_years": runtime.integer("backtest.minimum_period_years"),
            "maximum_period_years": runtime.integer("market.max_backtest_years"),
            "maximum_initial_capital": runtime.floating("backtest.maximum_initial_capital"),
            "maximum_fee_rate": runtime.floating("backtest.maximum_fee_rate"),
            "maximum_slippage_rate": runtime.floating("backtest.maximum_slippage_rate"),
            "minimum_probability_threshold": runtime.floating("backtest.minimum_probability_threshold"),
            "maximum_probability_threshold": runtime.floating("backtest.maximum_probability_threshold"),
        },
        "options": {
            "history_sources": ["MEXC", "BINANCE_SPOT_ARCHIVE"],
            "experiment_types": ["TARGET_STOP", "BTC_DIRECTION_RETURN"],
            "forecast_horizons_hours": runtime.list("forecast.horizons_hours"),
            "decision_timeframes": runtime.list("backtest.allowed_decision_timeframes"),
            "trend_timeframes": runtime.list("backtest.allowed_trend_timeframes"),
        },
        "ui": {
            "recent_jobs_limit": runtime.integer("frontend.recent_jobs_limit"),
            "job_poll_interval_ms": runtime.integer("frontend.job_poll_interval_ms"),
        },
    }
