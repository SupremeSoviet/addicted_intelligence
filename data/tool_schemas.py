from data.mock_values import (
    WEATHER,
    TIKTOK,
    CALCULATE,
    NOTES,
)

def get_weather(city: str) -> str:
    """
    Returns the current weather in a city.

    Args:
        city: name of the city
    """
    return WEATHER.get(city, "unknown city")

def create_note(note: str) -> str:
    """
    Returns whether note creation was successful or not

    Args:
        note: text of the note
    """

    return NOTES.get(note, "success!")

def whatch_tiktok() -> str:
    """
    Returns top tiktok from tiktok trends
    """

    return TIKTOK.get('top_tiktok', 'there is no connection with tiktok\'s servers')

def calculate_expression(expression: str) -> str:
    """
    Returns a calculated result of the input expression

    Args:
        expression: expression to calculate, do not include spaces
    """

    return CALCULATE.get(expression, "try another expression or ensure that expression does not contain spaces")