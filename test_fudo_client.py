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


# ──────────────────────────────────────────────────────────────
# Tests de zona horaria en los filtros de fecha hacia Fudo.
# Fudo espera timestamps UTC; el dia calendario del usuario es LOCAL.
# ──────────────────────────────────────────────────────────────

from contextlib import contextmanager

import main
from main import _date_filter


@contextmanager
def cliente_de(pais):
    """Activa un cliente con ese pais en el contextvar, como en una peticion real."""
    token = main._current_cliente_info.set({"id": 1, "pais": pais})
    try:
        yield
    finally:
        main._current_cliente_info.reset(token)


def rango(filtro: str) -> tuple[str, str]:
    """'and(gte.X,lte.Y)' -> (X, Y)"""
    assert filtro.startswith("and(gte.") and filtro.endswith(")")
    desde, hasta = filtro[len("and(gte."):-1].split(",lte.")
    return desde, hasta


def filtro_de(func, *args, **kwargs) -> str:
    """Ejecuta una funcion de consulta con _fudo_get mockeado y devuelve
    el valor de filter[createdAt] que se le paso a Fudo."""
    with patch.object(main, "_fudo_get") as mock_get:
        mock_get.return_value = {}
        func(*args, **kwargs)
    raw_query = mock_get.call_args.args[1]
    for parte in raw_query.split("&"):
        if parte.startswith("filter[createdAt]="):
            return parte[len("filter[createdAt]="):]
    raise AssertionError(f"sin filter[createdAt] en: {raw_query}")


def test_date_filter_chile_convierte_dia_local_a_utc():
    # 2026-09-03 en Chile es UTC-4 (el horario de verano empieza el 06/09).
    with cliente_de("Chile"):
        desde, hasta = rango(_date_filter("2026-09-03", "2026-09-03"))
    assert desde == "2026-09-03T04:00:00Z"
    assert hasta == "2026-09-04T03:59:59Z"


def test_venta_de_la_noche_anterior_queda_fuera_del_dia_chileno():
    # Bug real: closedAt 2026-09-03T00:08:13Z fue la noche del 02/09 en Chile
    # y se colaba en el reporte de "hoy" 03/09 inflando el total.
    venta_utc = "2026-09-03T00:08:13Z"
    with cliente_de("Chile"):
        desde_hoy, hasta_hoy = rango(_date_filter("2026-09-03", "2026-09-03"))
        desde_ayer, hasta_ayer = rango(_date_filter("2026-09-02", "2026-09-02"))

    assert not (desde_hoy <= venta_utc <= hasta_hoy), "no debe contar como hoy en Chile"
    assert desde_ayer <= venta_utc <= hasta_ayer, "debe contar como el dia 02/09 en Chile"


def test_date_filter_argentina_convierte_dia_local_a_utc():
    # Argentina es UTC-3 todo el año.
    with cliente_de("Argentina"):
        desde, hasta = rango(_date_filter("2026-09-03", "2026-09-03"))
    assert desde == "2026-09-03T03:00:00Z"
    assert hasta == "2026-09-04T02:59:59Z"


def test_date_filter_mexico_convierte_dia_local_a_utc():
    # Ciudad de Mexico es UTC-6 (sin horario de verano desde 2022).
    with cliente_de("Mexico"):
        desde, hasta = rango(_date_filter("2026-09-03", "2026-09-03"))
    assert desde == "2026-09-03T06:00:00Z"
    assert hasta == "2026-09-04T05:59:59Z"


@pytest.mark.parametrize(
    "pais,venta_utc",
    [
        # 23:59:59 hora local del 03/09, expresado en UTC segun cada offset.
        ("Chile", "2026-09-04T03:59:59Z"),
        ("Argentina", "2026-09-04T02:59:59Z"),
        ("Mexico", "2026-09-04T05:59:59Z"),
    ],
)
def test_venta_al_filo_de_las_2359_local_queda_dentro(pais, venta_utc):
    with cliente_de(pais):
        desde, hasta = rango(_date_filter("2026-09-03", "2026-09-03"))
    assert desde <= venta_utc <= hasta
    # Un segundo despues ya es el dia siguiente y debe quedar fuera.
    assert venta_utc == hasta


def test_pais_desconocido_o_vacio_cae_en_chile():
    with cliente_de("Narnia"):
        chile_por_defecto = _date_filter("2026-09-03", "2026-09-03")
    with cliente_de("Chile"):
        chile = _date_filter("2026-09-03", "2026-09-03")
    assert chile_por_defecto == chile

    token = main._current_cliente_info.set(None)
    try:
        sin_cliente = _date_filter("2026-09-03", "2026-09-03")
    finally:
        main._current_cliente_info.reset(token)
    assert sin_cliente == chile


def test_expenses_y_payments_usan_el_mismo_filtro_que_sales():
    # get_expenses y get_payments duplicaban el filtro en UTC puro (mismo bug).
    with cliente_de("Chile"):
        esperado = filtro_de(main.get_sales_report, "2026-09-01", "2026-09-03")
        assert filtro_de(main.get_expenses, "2026-09-01", "2026-09-03") == esperado
        assert filtro_de(main.get_payments, "2026-09-01", "2026-09-03") == esperado
        # Y las demas consultas por rango tambien.
        assert filtro_de(main.get_top_products, "2026-09-01", "2026-09-03") == esperado
        assert filtro_de(main.get_orders, "2026-09-01", "2026-09-03") == esperado
        assert filtro_de(main.get_deliveries_report, "2026-09-01", "2026-09-03") == esperado
