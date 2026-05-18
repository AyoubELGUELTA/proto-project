import pytest
import pandas as pd
import numpy as np
import json
import os
from unittest.mock import AsyncMock, MagicMock
from app.services.graph.graph_service import GraphService 
from app.core.data_model.text_units import TextUnit 

# Chemin vers notre snapshot de référence
SNAPSHOT_EXPECTED_PATH = "tests/data/mocks/output/final_data_output.json"

@pytest.mark.asyncio
async def test_run_pipeline_orchestration_snapshot():
    """
    Snapshot test pour figer l'orchestration globale de run_pipeline.
    Injecte des DataFrames conformes aux modèles EntityModel et RelationshipModel
    pour valider le re-mapping (Phase 3) sans lever de KeyError.
    """
    # 1. SETUP DES MOCKS : Isolation totale de l'infrastructure
    mock_extractor = AsyncMock()
    mock_summarizer = AsyncMock()
    mock_parser = MagicMock()
    mock_resolution_engine = AsyncMock()
    mock_store_manager = AsyncMock()
    mock_community_service = MagicMock()

    # 2. INJECTION DES DONNÉES EN ACCORD AVEC LES MODÈLES PYDANTIC
    # Simulation Phase 1 : Données brutes de l'extracteur/parser
    raw_entities = pd.DataFrame([
        {
            "id": "id-init-prophet", 
            "title": "Prophet", 
            "type": "SAHABI", 
            "slug": "PROPHET", 
            "category": "HUMAN", 
            "source_ids": ["chunk_0"]
        },
        {
            "id": "id-init-messenger", 
            "title": "Messenger", 
            "type": "SAHABI", 
            "slug": "MESSENGER", 
            "category": "HUMAN", 
            "source_ids": ["chunk_0_s5"]
        },
        {
            "id": "id-init-maymuna", 
            "title": "Maymuna", 
            "type": "SAHABIYYAT", 
            "slug": "MAYMUNA", 
            "category": "HUMAN", 
            "source_ids": ["chunk_0_s5"]
        }
    ])

    # Inclusions des colonnes critiques source_slug / target_slug
    raw_relationships = pd.DataFrame([
        {
            "id": "rel-1",
            "source_slug": "PROPHET", 
            "target_slug": "MAYMUNA", 
            "source_id": "id-init-prophet",
            "target_id": "id-init-maymuna",
            "description": "Visited each other",
            "weight": 1.0,
            "rank": 1,
            "source_ids": ["chunk_0"],       
            "attributes": {"importance": "high"} 
        },
        {
            "id": "rel-2",
            "source_slug": "MESSENGER", 
            "target_slug": "MAYMUNA", 
            "source_id": "id-init-messenger",
            "target_id": "id-init-maymuna",
            "description": "Spent time together",
            "weight": 1.0,
            "rank": 1,
            "source_ids": ["chunk_0_s5"],   
            "attributes": {}
        }
    ])
    mock_parser.to_dataframes.return_value = (raw_entities, raw_relationships)

    # Simulation Phase 2 : Le ResolutionEngine fusionne Messenger dans Prophet
    resolved_entities = pd.DataFrame([
        {
            "id": "id-init-prophet", 
            "title": "Prophet", 
            "type": "SAHABI", 
            "slug": "PROPHET", 
            "category": "HUMAN", 
            "source_ids": ["chunk_0", "chunk_0_s5"]
        },
        {
            "id": "id-init-maymuna", 
            "title": "Maymuna", 
            "type": "SAHABIYYAT", 
            "slug": "MAYMUNA", 
            "category": "HUMAN", 
            "source_ids": ["chunk_0_s5"]
        }
    ])
    
    global_mapping_mock = {
        "id-init-prophet": "id-init-prophet",
        "id-init-messenger": "id-init-prophet",
        "id-init-maymuna": "id-init-maymuna"
    }
    mock_resolution_engine.run.return_value = (resolved_entities, global_mapping_mock)
    
    # Simulation Phase 4 : Le summarizer bypass
    mock_summarizer.summarize_all.side_effect = lambda entities_df, relationships_df: (entities_df, relationships_df)

    # 3. INSTANCIATION DE L'ORCHESTRATEUR
    service = GraphService(
        extractor=mock_extractor,
        summarizer=mock_summarizer,
        parser=mock_parser,
        resolution_engine=mock_resolution_engine,
        store_manager=mock_store_manager,
        community_service=mock_community_service
    )

    text_units = [TextUnit(id="chunk_0", text="..."), TextUnit(id="chunk_0_s5", text="...")]

    # Court-circuitage propre du workflow global des communautés (Phase 6)
    import app.services.graph.graph_service as gs
    mock_run_communities = AsyncMock()
    gs.run_create_communities_workflow = mock_run_communities

    # 4. EXÉCUTION DU PIPELINE
    entities_result, rels_result = await service.run_pipeline(
        text_units=text_units,
        domain_context="Islamic History",
        persist=True 
    )

    # 5. VERIFICATION DES VALVES INFRASTRUCTURE
    assert mock_store_manager.save_graph.called, "❌ Phase 5 : Le graphe n'a pas été persisté dans la base."
    assert mock_run_communities.called, "❌ Phase 6 : Le workflow des communautés n'a pas été déclenché."

    # 6. NORMALISATION ET SÉRIALISATION DU SNAPSHOT
    entities_json = entities_result.replace({np.nan: None}).to_dict(orient="records")
    rels_json = rels_result.replace({np.nan: None}).to_dict(orient="records")

    # On trie uniquement les lignes (les dictionnaires) pour que l'index de liste corresponde
    entities_json = sorted(entities_json, key=lambda x: x.get("id", ""))
    rels_json = sorted(rels_json, key=lambda x: (x.get("source_id", ""), x.get("target_id", "")))

    current_snapshot = {
        "entities": entities_json,
        "relationships": rels_json
    }

    # Génération automatique du dossier et du fichier Master s'il est absent
    if not os.path.exists(SNAPSHOT_EXPECTED_PATH):
        os.makedirs(os.path.dirname(SNAPSHOT_EXPECTED_PATH), exist_ok=True)
        with open(SNAPSHOT_EXPECTED_PATH, "w", encoding="utf-8") as f:
            json.dump(current_snapshot, f, indent=4, ensure_ascii=False)
        pytest.skip(f"📝 Master Snapshot initialisé dans {SNAPSHOT_EXPECTED_PATH}. Relancer pour valider.")

    # Chargement du Master
    with open(SNAPSHOT_EXPECTED_PATH, "r", encoding="utf-8") as f:
        expected_snapshot = json.load(f)

   # 7. SÉCURITÉS ET VERDICT (ASSERTIONS INTELLIGENTES & LAXISTES SUR L'ORDRE)
    assert len(current_snapshot["entities"]) == len(expected_snapshot["entities"]), "❌ Le nombre d'entités uniques a changé !"
    assert len(current_snapshot["relationships"]) == len(expected_snapshot["relationships"]), "❌ Le nombre de relations a changé !"

    # A. Validation stricte des entités
    assert current_snapshot["entities"] == expected_snapshot["entities"], "❌ Dérive détectée dans les entités !"

    # B. Validation laxiste des relations (ordre des chaînes et listes ignoré)
    for current_rel, expected_rel in zip(current_snapshot["relationships"], expected_snapshot["relationships"]):
        for key in current_rel.keys():
            if key == "description":
                # Extraction des descriptions sous forme de set
                current_desc_set = set(p.strip() for p in current_rel.get("description", "").split("|") if p.strip())
                expected_desc_set = set(p.strip() for p in expected_rel.get("description", "").split("|") if p.strip())
                assert current_desc_set == expected_desc_set, f"❌ Contenu de description asymétrique !"
            
            elif isinstance(current_rel[key], list):
                # Si le champ est une liste (ex: source_ids), on ignore l'ordre des éléments
                assert set(current_rel[key]) == set(expected_rel.get(key, [])), f"❌ Différence de contenu détectée sur la liste '{key}'"
            
            else:
                # Validation stricte pour le reste (valeurs uniques comme id, weight, rank)
                assert current_rel[key] == expected_rel.get(key), f"❌ Différence détectée sur le champ '{key}'"