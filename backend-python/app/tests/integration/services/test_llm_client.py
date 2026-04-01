import pytest
from unittest.mock import AsyncMock, patch
from langchain_core.messages import SystemMessage, HumanMessage

@pytest.mark.asyncio
async def test_llm_client_flow(mock_llm_service):
    service = mock_llm_service
    
    # 1. On vide le cache pour être SÛR de déclencher l'appel réseau (le Mock)
    service.cache.client.flushdb() 
    
    # 2. On configure le mock
    service.client._execute_with_retry = AsyncMock(return_value="('entity'<|>A<|>B)")
    
    # 3. Appel
    result = await service.extract_tuples("Bonjour Système", "Bonjour Utilisateur")
    
    # 4. On vérifie les appels
    # On utilise .called d'abord pour être sûr
    assert service.client._execute_with_retry.called
    
    # Récupération sécurisée des arguments
    args = service.client._execute_with_retry.call_args[0]
    sent_messages = args[0] # Le premier argument de _execute_with_retry est la liste de messages
    
    assert sent_messages[0].content == "Bonjour Système"
    assert result == [["entity", "A", "B"]]
    assert service.client._execute_with_retry.called


@pytest.mark.asyncio
async def test_llm_client_cache_hit(mock_llm_service):
    """Vérifie que si la donnée est en cache, on n'appelle JAMAIS l'API."""
    
    service = mock_llm_service
    # On mock l'appel pour vérifier qu'il n'est PAS appelé
    service.client._execute_with_retry = AsyncMock()
    
    # Utiliser les mêmes objets que dans service.extract_tuples
    messages = [
        SystemMessage(content="sys"),
        HumanMessage(content="user")
    ]
    
    # On force la valeur en cache avec la bonne clé
    service.cache.set(messages, "('cache'<|>HIT)")
    
    # Appel
    result = await service.extract_tuples("sys", "user")
    
    # ASSERTIONS
    assert result == [["cache", "HIT"]]
    service.client._execute_with_retry.assert_not_called()