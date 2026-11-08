from fastapi.testclient import TestClient

from crypto_paper_trader_ml_api.app import app


def test_health_and_database_configuration() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        configuration = client.get("/v1/ml-configuration")
        settings = client.get("/v1/ml-settings")

    assert response.status_code == 200
    assert response.json()["service"] == "crypto-paper-trader-ml-api"
    assert response.json()["parameter_source"] == "DATABASE"
    assert configuration.status_code == 200
    assert configuration.json()["source"] == "DATABASE"
    assert configuration.json()["defaults"]["symbol"]
    assert configuration.json()["options"]["decision_timeframes"]
    assert configuration.json()["options"]["history_sources"] == ["MEXC", "BINANCE_SPOT_ARCHIVE"]
    assert configuration.json()["defaults"]["history_source"] in {"MEXC", "BINANCE_SPOT_ARCHIVE"}
    assert configuration.json()["defaults"]["experiment_type"] in {"TARGET_STOP", "BTC_DIRECTION_RETURN"}
    assert configuration.json()["options"]["experiment_types"] == ["TARGET_STOP", "BTC_DIRECTION_RETURN"]
    assert set(configuration.json()["options"]["forecast_horizons_hours"])
    visible = {item["key"]: item for item in settings.json()["parameters"]}
    assert visible["forecast.horizons_hours"]["value"] == configuration.json()["options"]["forecast_horizons_hours"]
    assert 2 <= visible["forecast.maximum_folds"]["value"] <= 6
    assert visible["backtest.default_experiment_type"]["value"] == configuration.json()["defaults"]["experiment_type"]
    assert configuration.json()["ui"]["job_poll_interval_ms"] > 0
    assert settings.status_code == 200
    assert len(settings.json()["parameters"]) > 50


def test_railway_frontend_origin_is_allowed_without_restart() -> None:
    origin = "https://cryptopapertraderfront-production.up.railway.app"
    with TestClient(app) as client:
        preflight = client.options(
            "/v1/ml-backtests",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        health = client.get("/health", headers={"Origin": origin})

    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == origin
    assert health.status_code == 200
    assert health.headers["access-control-allow-origin"] == origin
