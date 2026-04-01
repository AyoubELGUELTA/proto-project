import pytest
import asyncio
from unittest.mock import MagicMock
from app.services.llm.service import LLMService

@pytest.fixture(scope="session")
def event_loop():
    """Garantit un seul loop asyncio pour toute la session de test."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_llm_service():
    """
    Fournit une instance de LLMService avec un client LLM mocké.
    Évite les appels API réels et les coûts associés pendant les tests.
    """
    service = LLMService(model_name="gpt-4o-mini")
    # On mocke l'appel réseau pur
    service.client._execute_with_retry = MagicMock()
    return service