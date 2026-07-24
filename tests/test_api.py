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
    assert configuration.json()["ui"]["job_poll_interval_ms"] > 0
    assert settings.status_code == 200
    assert len(settings.json()["parameters"]) > 50
