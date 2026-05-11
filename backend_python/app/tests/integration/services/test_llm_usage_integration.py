import pytest
from app.services.llm.factory import LLMFactory
import time
@pytest.mark.asyncio
async def test_real_llm_call_updates_tracker():
    """
    Test 'Live' : On appelle vraiment OpenAI pour vérifier que 
    les métadonnées de consommation sont correctement capturées.
    """
    # 1. Initialisation via la Factory (qui utilise tes settings.py tout neufs)
    # Le tracker est normalement attaché statiquement à la Factory ou au Service
    llm_service = LLMFactory.get_service()
    tracker = LLMFactory.get_tracker()
    
    # On reset le tracker pour repartir de zéro
    tracker.usage.total_tokens = 0
    tracker.usage.total_cost = 0.0
    
    # 2. Exécution d'un appel réel
    
    prompt = f"Dis 'Hello' en un mot. Timestamp: {time.time()}" #To be sure the prompt was not cached (then the answer is costless, the test is obsolete)
    response = await llm_service.ask_text(
        system_prompt="Tu es un assistant minimaliste.",
        user_prompt=prompt
    )
    
    # 3. Vérifications
    print(f"\n🤖 Réponse LLM: {response}")
    print(f"📊 Stats après appel: {tracker.get_report()}")
    
    # On vérifie que les tokens ne sont plus à zéro
    assert tracker.usage.total_tokens > 0, "Le nombre de tokens devrait avoir augmenté"
    assert tracker.usage.total_cost > 0, "Le coût total devrait être supérieur à 0"
    
    # Vérification spécifique pour gpt-4o-mini (notre modèle par défaut)
    # Les tokens d'entrée pour un prompt court + system prompt tournent souvent autour de 20-30
    assert 10 < tracker.usage.total_tokens < 100