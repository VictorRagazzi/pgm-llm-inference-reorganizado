import httpx
import pytest
import requests

from pgm_llm_inference.core.config import InferenceConfig


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch):
    """Never read local credentials or contact providers in the test suite."""
    monkeypatch.setitem(InferenceConfig.model_config, "env_file", None)
    monkeypatch.setitem(InferenceConfig.model_config, "env_prefix", "PGM_TEST_")

    def reject_network(*args, **kwargs):
        raise AssertionError("Tests must mock HTTP requests")

    monkeypatch.setattr(httpx.Client, "send", reject_network)
    monkeypatch.setattr(httpx.AsyncClient, "send", reject_network)
    monkeypatch.setattr(requests.Session, "send", reject_network)
