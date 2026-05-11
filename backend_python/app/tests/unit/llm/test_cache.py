# tests/unit/llm/test_cache.py
import pytest
from unittest.mock import MagicMock
from app.services.llm.cache import LLMCache

def test_cache_key_generation():
    """Vérifie que deux messages différents produisent des clés différentes."""
    cache = LLMCache()
    
    msg1 = [{"role": "user", "content": "Hello"}]
    msg2 = [{"role": "user", "content": "Hi"}]
    
    key1 = cache._generate_key(msg1)
    key2 = cache._generate_key(msg2)
    
    assert key1.startswith("llm_cache:")
    assert key1 != key2
    # Vérifie que la clé est toujours la même pour le même message
    assert key1 == cache._generate_key(msg1)

def test_cache_logic_with_mock():
    """Vérifie la logique get/set en simulant Redis."""
    cache = LLMCache()
    # On remplace le client redis réel par un mock
    cache.client = MagicMock()
    
    messages = [{"content": "test"}]
    cache.set(messages, "response")
    
    # Vérifie que Redis a bien reçu la commande 'setex' (set with expiry)
    cache.client.setex.assert_called_once()