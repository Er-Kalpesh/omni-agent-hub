"""Unit tests for OmniAgent Tools and Calculator Functions."""

import pytest
from agent.tools import (
    calculate_expression,
    get_weather_forecast,
    fetch_financial_stock_quote,
    execute_python_code_sandbox
)

def test_calculate_expression_valid():
    res = calculate_expression("sqrt(144) * 5")
    assert res["status"] == "success"
    assert res["result"] == 60.0

def test_calculate_expression_invalid():
    res = calculate_expression("invalid_func()")
    assert res["status"] == "error"

def test_weather_forecast():
    res = get_weather_forecast("Tokyo")
    assert res["status"] == "success"
    assert res["data"]["location"] == "Tokyo"

def test_stock_quote():
    res = fetch_financial_stock_quote("GOOGL")
    assert res["status"] == "success"
    assert res["ticker"] == "GOOGL"

def test_python_sandbox():
    code = "print(2 + 2)"
    res = execute_python_code_sandbox(code)
    assert res["status"] == "success"
    assert "4" in res["output"]
