import pytest
import redis
from app.services.llm.cache import LLMCache

def test_redis_connection_live():
    """Vérifie que le service peut réellement parler à Redis."""
    cache = LLMCache()
    try:
        # On tente un PING
        response = cache.client.ping()
        assert response is True, "Redis n'a pas répondu PONG au PING"
    except redis.ConnectionError:
        pytest.fail("❌ Redis n'est pas accessible. Vérifie ton container Docker.")

def test_redis_persistence():
    """Vérifie que les données survivent à une ré-instanciation du service."""
    cache = LLMCache()
    test_key = [{"content": "persistence_test"}]
    test_val = "verified"
    
    cache.set(test_key, test_val)
    
    # On crée une nouvelle instance pour simuler un redémarrage
    new_cache = LLMCache()
    assert new_cache.get(test_key) == test_val