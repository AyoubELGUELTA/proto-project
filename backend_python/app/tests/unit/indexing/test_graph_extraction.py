import pytest
import pandas as pd
from unittest.mock import AsyncMock, MagicMock
from app.indexing.operations.graph.graph_extractor import EntityAndRelationExtractor
from app.indexing.operations.graph.extract_graph import extract_graph # THIS FUNCTION WAS DELETED, HAVE TO RE ARRANGE THIS FILE TO MAKE IT USE ONLY THE EXTrACTOR CLASS
from app.models.config import ExtractGraphConfig
from app.models.domain import SiraEntityType
from app.tests.mocks.llm_outputs import MOCK_SIRA_EXTRACTION_CHUNK_1, MOCK_SIRA_EXTRACTION_CHUNK_2
@pytest.mark.asyncio
async def test_sira_graph_extraction_logic():
    # 1. Setup Config & Mock LLM
    config = ExtractGraphConfig(
        entity_types=[e.value for e in SiraEntityType],
        max_gleanings=0 # On désactive le gleaning pour le test unitaire simple
    )
    
    mock_llm = MagicMock()
    mock_llm.ask = AsyncMock()
    # On simule les réponses successives pour les deux chunks
    mock_llm.ask.side_effect = [MOCK_SIRA_EXTRACTION_CHUNK_1, MOCK_SIRA_EXTRACTION_CHUNK_2]

    extractor = EntityAndRelationExtractor(llm_service=mock_llm, config=config)

    # 2. Mock TextUnits (Fiche d'identité incluse)
    mock_metadata = {
        "TITLE": "Sira Overview",
        "SUBJECT_MATTER": "Prophetic biography and early battles",
        "CORE_ENTITIES": ["Muhammad", "Madinah", "Badr"]
    }

    text_units = [
        {"id": "chunk_1", "text": "The Prophet migrated to Madinah.", "metadata": mock_metadata},
        {"id": "chunk_2", "text": "He led the army at the Battle of Badr.", "metadata": mock_metadata}
    ]

    # 3. Execution du workflow d'extraction
    entities_df, relations_df = await extract_graph(text_units, extractor)

    # --- ASSERTIONS ---

    # A. Test du Merging des Entités
    # MUHAMMAD doit apparaître UNE SEULE fois car il est fusionné
    muhammad = entities_df[entities_df["title"] == "MUHAMMAD"]
    assert len(muhammad) == 1
    assert muhammad.iloc[0]["frequency"] == 2
    # Il doit avoir collecté les deux descriptions dans une liste
    assert len(muhammad.iloc[0]["description"]) == 2

    # B. Test de l'existence des autres entités
    assert "MADINAH" in entities_df["title"].values
    assert "BATTLE OF BADR" in entities_df["title"].values

    # C. Test du filtrage des Orphelins (L'élément clé de Microsoft)
    # UNKNOWN_PERSON n'a pas d'entrée entity, la relation doit avoir disparu
    assert "UNKNOWN_PERSON" not in relations_df["target"].values
    # Il ne doit rester que 2 relations valides
    assert len(relations_df) == 2

    print("\n✅ Test Graph Extraction: SUCCESS")
    print(f"Total Entities merged: {len(entities_df)}")
    print(f"Total Relations filtered: {len(relations_df)}")