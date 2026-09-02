"""
Tests de FudoClient.get(): reintentos con backoff ante fallas transitorias.
Uso: pytest test_fudo_client.py
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from main import FudoClient


def make_client() -> FudoClient:
    client = FudoClient(api_key="k", api_secret="s")
    # Evita la llamada real de autenticación: siempre hay un token "vigente".
    client._get_token = MagicMock(return_value="dummy-token")
    return client


def make_response(status_code: int, payload: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload or {}
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(f"{status_code} error")
    else:
        resp.raise_for_status.side_effect = None
    return resp


@patch("main.time.sleep", return_value=None)
@patch("main.requests.get")
def test_get_retries_after_timeout_then_succeeds(mock_get, mock_sleep):
    client = make_client()
    mock_get.side_effect = [
        requests.exceptions.Timeout("timed out"),
        make_response(200, {"ok": True}),
    ]

    result = client.get("/sales")

    assert result == {"ok": True}
    assert mock_get.call_count == 2
    mock_sleep.assert_called_once_with(1)


@patch("main.time.sleep", return_value=None)
@patch("main.requests.get")
def test_get_retries_after_503_then_succeeds(mock_get, mock_sleep):
    client = make_client()
    mock_get.side_effect = [
        make_response(503),
        make_response(200, {"ok": True}),
    ]

    result = client.get("/sales")

    assert result == {"ok": True}
    assert mock_get.call_count == 2
    mock_sleep.assert_called_once_with(1)


@patch("main.time.sleep", return_value=None)
@patch("main.requests.get")
def test_get_raises_runtime_error_after_repeated_timeouts(mock_get, mock_sleep):
    client = make_client()
    mock_get.side_effect = [
        requests.exceptions.Timeout("timed out"),
        requests.exceptions.Timeout("timed out"),
        requests.exceptions.Timeout("timed out"),
    ]

    with pytest.raises(RuntimeError, match="Fudo: fallo de red tras reintentos"):
        client.get("/sales")

    assert mock_get.call_count == 3
