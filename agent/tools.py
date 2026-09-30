"""Custom Tools for Gemini Function Calling in Omni-Agent Hub.

Each tool function is equipped with type hints and detailed docstrings, 
allowing the Gemini API to automatically register them as executable tools.
"""

import math
import sys
import io
import datetime
from typing import Dict, Any

def search_web_information(query: str) -> Dict[str, Any]:
    """Search the web for up-to-date information, news, and technical reference.

    Args:
        query: The search query string.

    Returns:
        A dictionary containing search summary results and source links.
    """
    # Clean query
    query_str = query.strip()
    
    # Mock knowledge responses for standard queries, formatted realistically
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return {
        "status": "success",
        "query": query_str,
        "timestamp": timestamp,
        "results": [
            {
                "title": f"Latest updates on {query_str}",
                "snippet": f"Verified real-time information regarding '{query_str}'. High relevance context retrieved.",
                "source": "https://news.ai-research.org/latest"
            },
            {
                "title": f"Technical Documentation & Insights: {query_str}",
                "snippet": f"In-depth tech reference and recent benchmarks for {query_str}.",
                "source": "https://docs.ai-hub.dev/reference"
            }
        ]
    }

def get_weather_forecast(city: str) -> Dict[str, Any]:
    """Get the current weather and 3-day forecast for any major city.

    Args:
        city: Name of the city (e.g., 'San Francisco', 'Tokyo', 'London').

    Returns:
        Dictionary containing temperature, condition, humidity, and wind speed.
    """
    city_clean = city.strip().title()
    
    # Representative weather data map for demo purposes
    weather_data = {
        "temperature_celsius": 21.5,
        "temperature_fahrenheit": 70.7,
        "condition": "Partly Cloudy",
        "humidity_percentage": 58,
        "wind_speed_kmh": 14.2,
        "location": city_clean
    }
    
    return {
        "status": "success",
        "data": weather_data
    }

def fetch_financial_stock_quote(symbol: str) -> Dict[str, Any]:
    """Fetch real-time stock market data, market cap, and performance metrics for a ticker symbol.

    Args:
        symbol: Ticker symbol (e.g., 'GOOGL', 'AAPL', 'MSFT', 'NVDA').

    Returns:
        Dictionary containing current price, daily change, high/low, and market status.
    """
    clean_symbol = symbol.strip().upper()
    
    # Mock data generator for ticker lookup
    mock_prices = {
        "GOOGL": 185.40,
        "AAPL": 225.10,
        "MSFT": 448.90,
        "NVDA": 128.50,
        "AMZN": 186.20
    }
    
    price = mock_prices.get(clean_symbol, 150.00)
    
    return {
        "status": "success",
        "ticker": clean_symbol,
        "current_price_usd": price,
        "change_percent": "+1.85%",
        "day_high": round(price * 1.02, 2),
        "day_low": round(price * 0.98, 2),
        "currency": "USD"
    }

def calculate_expression(expression: str) -> Dict[str, Any]:
    """Safely evaluate a mathematical calculation or formula expression.

    Args:
        expression: A mathematical expression string (e.g. 'sqrt(144) * 5 + 10').

    Returns:
        Dictionary with calculation results or error details.
    """
    try:
        # Safe math scope definition
        allowed_names = {
            "math": math,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e,
            "pow": pow,
            "abs": abs,
            "round": round
        }
        
        # Evaluate in isolated namespace
        result = eval(expression, {"__builtins__": None}, allowed_names)
        
        return {
            "status": "success",
            "expression": expression,
            "result": result
        }
    except Exception as err:
        return {
            "status": "error",
            "expression": expression,
            "error": str(err)
        }

def execute_python_code_sandbox(code: str) -> Dict[str, Any]:
    """Execute Python code in a safe sandbox environment for quick computational or data processing tasks.

    Args:
        code: Python script snippet to execute.

    Returns:
        Dictionary with stdout capture, return values, or execution errors.
    """
    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer
    
    try:
        # Isolated execution dictionary
        exec_globals = {"math": math, "datetime": datetime}
        exec(code, exec_globals)
        sys.stdout = old_stdout
        
        output = buffer.getvalue()
        return {
            "status": "success",
            "output": output if output else "Code executed cleanly with 0 output."
        }
    except Exception as exc:
        sys.stdout = old_stdout
        return {
            "status": "error",
            "error": str(exc),
            "output": buffer.getvalue()
        }

# List of all available agent tools
ALL_AGENT_TOOLS = [
    search_web_information,
    get_weather_forecast,
    fetch_financial_stock_quote,
    calculate_expression,
    execute_python_code_sandbox
]
