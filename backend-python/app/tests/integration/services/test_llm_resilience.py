import pytest
from app.services.llm.service import LLMService
from unittest.mock import patch
import redis

@pytest.mark.asyncio
async def test_service_works_even_if_redis_is_down():
    """
    Test de résilience : Si Redis crash, le service doit quand même 
    répondre en appelant le LLM directement.
    """
    service = LLMService()
    
    # On simule une erreur de connexion Redis
    with patch.object(service.cache.client, 'get', side_effect=redis.ConnectionError):
        # On mock l'appel LLM pour ne pas payer
        with patch.object(service.client, '_execute_with_retry', return_value="('result')"):
            result = await service.ask_json("sys", "user")
            
            # Le service doit avoir fonctionné malgré l'erreur Redis
            assert result is not None
            print("✅ Résilience validée : Le service survit à une panne Cache.")