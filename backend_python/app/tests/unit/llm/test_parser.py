import pytest
from app.services.llm.parser import LLMParser

def test_to_tuples_valid_format():
    """Vérifie que le format Microsoft est correctement transformé en liste."""
    raw_input = '("entity"<|>MAYMUNA<|>PERSON) ## ("relationship"<|>MAYMUNA<|>MARRIED_TO<|>PROPHET)'
    expected = [
        ["entity", "MAYMUNA", "PERSON"],
        ["relationship", "MAYMUNA", "MARRIED_TO", "PROPHET"]
    ]
    assert LLMParser.to_tuples(raw_input) == expected

def test_to_json_with_markdown():
    """Vérifie que le parser nettoie bien les balises markdown JSON."""
    raw_input = """```json
    {
        "entity": "Maymuna",
        "type": "Person"
    }
    ```"""
    expected = {"entity": "Maymuna", "type": "Person"}
    result = LLMParser.to_json(raw_input)
    
    assert result == expected
    assert isinstance(result, dict)