from __future__ import annotations

from crypto_paper_trader_ml_api.database import initialize_database
from crypto_paper_trader_ml_api.settings_service import RuntimeParameters, get_runtime_parameters


def default_parameters(**overrides):  # type: ignore[no-untyped-def]
    initialize_database()
    values = get_runtime_parameters().snapshot()
    values.update(overrides)
    return RuntimeParameters(values)
