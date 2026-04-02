import pytest
import pandas as pd
from unittest.mock import AsyncMock, MagicMock

from app.indexing.operations.graph.summarize_extractor import SummarizeExtractor
from app.indexing.operations.graph.summarize_descriptions import summarize_descriptions
from app.tests.mocks.graph_data import MOCK_ENTITY_SUMMARIZATION_INPUT, MOCK_RELATIONSHIP_SUMMARIZATION_INPUT
from app.tests.mocks.llm_outputs import MOCK_SUMMARIZER_LLM_RESPONSE

@pytest.mark.asyncio
async def test_summarize_descriptions_logic():
    
    entities_df = pd.DataFrame(MOCK_ENTITY_SUMMARIZATION_INPUT)
    relationships_df = pd.DataFrame(MOCK_RELATIONSHIP_SUMMARIZATION_INPUT)

    mock_llm = MagicMock()
    mock_llm.ask = AsyncMock(return_value=MOCK_SUMMARIZER_LLM_RESPONSE)

    extractor = SummarizeExtractor(
        llm_client=mock_llm,
        prompt_template="Entity: {entity_name}, List: {description_list}, Max: {max_length}",
        max_input_tokens=1000,
        max_summary_length=100
    )

    res_entities_df, res_rel_df = await summarize_descriptions(
        entities_df=entities_df,
        relationships_df=relationships_df,
        extractor=extractor,
        num_threads=2
    )

    
    # A. Vérification de la transformation (List[str] -> str)
    assert isinstance(res_entities_df.iloc[0]["description"], str)
    assert res_entities_df.iloc[0]["description"] == MOCK_SUMMARIZER_LLM_RESPONSE
    
    # B. Vérification du nombre d'appels LLM
    # (Nombre d'entités unique + Nombre de relations unique)
    expected_calls = len(MOCK_ENTITY_SUMMARIZATION_INPUT) + len(MOCK_RELATIONSHIP_SUMMARIZATION_INPUT)
    assert mock_llm.ask.call_count == expected_calls
    
    # C. Vérification des colonnes
    assert "title" in res_entities_df.columns
    assert "source" in res_rel_df.columns

    print(f"\n🟢 Summarization Logic Test: SUCCESS ({expected_calls} LLM calls simulated)")