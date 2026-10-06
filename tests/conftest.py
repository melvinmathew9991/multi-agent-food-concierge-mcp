import pytest

from food_concierge.config import Settings, get_settings
from food_concierge.models.usage import get_usage_ledger


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    # Tests must not depend on the developer's shell: clear every variable Settings reads.
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_usage_ledger() -> None:
    # Call caps are per process; each test starts with none of its providers' quota used.
    get_usage_ledger().reset()
