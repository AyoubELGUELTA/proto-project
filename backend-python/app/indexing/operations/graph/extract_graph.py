# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

import logging
import pandas as pd
from typing import List, Tuple, Dict, Any
from .graph_extractor import GraphExtractor
from .utils import filter_orphan_relationships

logger = logging.getLogger(__name__)

async def extract_graph(
    text_units: List[Dict[str, Any]], # Liste de tes TextUnits (id, text, metadata)
    extractor: GraphExtractor,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orchestre l'extraction sur tous les chunks et fusionne les résultats.
    """
    all_entity_dfs = []
    all_relation_dfs = []

    for unit in text_units:
        # 1. Extraction brute via le LLM (avec Gleaning)
        ent_df, rel_df = await extractor(
            text=unit["text"],
            metadata=unit["metadata"],
            source_id=unit["id"]
        )
        
        all_entity_dfs.append(ent_df)
        all_relation_dfs.append(rel_df)

    if not all_entity_dfs:
        return pd.DataFrame(), pd.DataFrame()

    # 2. Fusion des Entités (Merging) : on regroupe par Nom et Type pour ne pas avoir de doublons

    entities = pd.concat(all_entity_dfs, ignore_index=True)
    entities = (
        entities.groupby(["title", "type"], sort=False)
        .agg({
            "description": list,  # On garde toutes les versions pour le futur Summary
            "source_id": list,    # Provenance
        })
        .reset_index()
    )
    # Ajout de la fréquence (nombre de fois où l'entité a été vue)
    entities["frequency"] = entities["source_id"].apply(len)

    # 3. Fusion des Relations
    relationships = pd.concat(all_relation_dfs, ignore_index=True)
    relationships = (
        relationships.groupby(["source", "target"], sort=False)
        .agg({
            "description": list,
            "weight": "sum",
            "source_id": list
        })
        .reset_index()
    )

    # 4. Nettoyage Strict (Le garde-fou MC)
    relationships = filter_orphan_relationships(relationships, entities)

    return entities, relationships