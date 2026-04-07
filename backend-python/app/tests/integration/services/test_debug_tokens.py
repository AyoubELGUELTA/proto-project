import pytest
from langchain_openai import ChatOpenAI
from app.core.settings import settings

@pytest.mark.asyncio
async def test_debug_openai_metadata_structure():
    """
    Test de diagnostic pur pour inspecter la carrosserie de la réponse OpenAI.
    """
    print("\n\n--- 🛰️  CONNEXION OPENAI EN COURS ---")
    
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        openai_api_key=settings.openai_api_key
    )
    
    # On fait un appel minuscule pour ne pas consommer
    response = await llm.ainvoke("Dis 'OK'")

    
    print("\n--- 🔍 RÉSULTATS DE L'INSPECTION ---")
    
    # 1. Le nouveau standard (LangChain 0.2+)
    usage_metadata = getattr(response, 'usage_metadata', None)
    print(f"STRICT USAGE METADATA : {usage_metadata}")
    
    # 2. L'ancien dictionnaire (Legacy)
    token_usage = response.response_metadata.get('token_usage', '❌ Vide')
    print(f"LEGACY TOKEN USAGE : {token_usage}")
    
    # 3. La structure brute pour être sûr de ne rien rater
    print(f"FULL METADATA DICT : {response.response_metadata}")
    print("\n--- END OF DIAGNOSTIC ---\n")

    assert response.content is not None