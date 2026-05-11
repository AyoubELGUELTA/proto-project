import pytest
from app.core.settings import settings
from app.services.llm.factory import LLMFactory
from app.indexing.operations.graph.summarize_extractor import SummarizeExtractor
from app.core.prompts.graph_prompts import SUMMARIZE_PROMPT
from app.core.config.llm_config import SUMMARIZATION_LLM_CONFIG
from app.core.config.graph_config import SummarizationConfig
from app.tests.mocks.graph_data import MOCK_ENTITY_DESCRIPTIONS

@pytest.mark.asyncio
async def test_summarize_sira_real_call():
    """
    Test d'intégration réel : Valide que le prompt Microsoft + notre Factory
    produisent un résumé de qualité pour Khadija (r.a).
    """
    if not settings.openai_api_key:
        pytest.skip("Skipping: OpenAI API Key non trouvée dans les settings.")

    # 2. Utilisation de la Factory pour créer le client avec le preset 'Summarization'
    # La factory s'occupe d'injecter le Tracker et le Cache Redis automatiquement
    llm_client = LLMFactory.create_client(config=SUMMARIZATION_LLM_CONFIG)

    # 3. Utilisation de la config métier pour le Summarizer
    graph_config = SummarizationConfig(
        max_summary_length=150, 
        max_input_tokens=2000
    )

    # 4. Instanciation de l'expert (L'artisan reçoit son outil + sa règle)
    extractor = SummarizeExtractor(
        llm_client=llm_client,
        config=graph_config,
        prompt_template=SUMMARIZE_PROMPT
    )

    entity_id = "KHADIJA BINT KHUWAYLID"
    
    print(f"\n🚀 [INTEGRATION] Appel LLM réel pour : {entity_id}...")
    
    # 5. Exécution
    final_summary = await extractor(id=entity_id, descriptions=MOCK_ENTITY_DESCRIPTIONS)

    # 6. Output visuel pour validation humaine
    print(f"\n--- RÉSULTAT DU SUMMARIZER ---")
    print(final_summary)
    print("------------------------------")

    # 7. Assertions qualitatives
    assert len(final_summary) > 100, "Le résumé est trop court."
    assert any(x in final_summary for x in ["Prophet", "Muhammad", "Islam"]), "Mots clés manquants."
    
    # Optionnel : Vérifier si le tracker a bien enregistré l'appel
    report = LLMFactory._tracker.get_report()
    print(f"\n📊 Consommation du test : {report}")