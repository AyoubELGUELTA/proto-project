# Copyright (C) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Utility functions for graph extraction operations."""


import pandas as pd


import logging
logger = logging.getLogger(__name__)



def filter_orphan_relationships(
    relationships: pd.DataFrame,
    entities: pd.DataFrame,
) -> pd.DataFrame:
    if relationships.empty or entities.empty:
        return relationships.iloc[0:0].reset_index(drop=True)

    # 1. On normalise les titres existants pour la comparaison
    entity_titles = set(entities["title"])
    
    # 2. Identification des orphelins (pour le debug)
    all_mentioned = set(relationships["source"]).union(set(relationships["target"]))
    orphans = all_mentioned - entity_titles

    if orphans:
        logger.warning(f"🕵️ Found {len(orphans)} entities mentioned in relations but missing in entity list.")
        logger.debug(f"🔍 List of orphans to be 'mocked': {list(orphans)}")

    # 3. L'approche de "Guérison" : 
    # Au lieu de filtrer ici (ce qui supprimerait les lignes), 
    # on va s'assurer que le mask est utilisé pour informer le système.
    
    # Pour l'instant, gardons le filtrage mais avec un LOG ULTRA PRÉCIS 
    # pour que tu vois SI c'est un problème de casse ou d'absence totale.
    
    mask_source = relationships["source"].isin(entity_titles)
    mask_target = relationships["target"].isin(entity_titles)
    
    # Debug des noms qui posent problème
    if not mask_source.all():
        bad_sources = relationships[~mask_source]["source"].unique()
        logger.debug(f"❌ Missing Sources: {list(bad_sources)}")
    
    if not mask_target.all():
        bad_targets = relationships[~mask_target]["target"].unique()
        logger.debug(f"❌ Missing Targets: {list(bad_targets)}")

    before_count = len(relationships)
    filtered = relationships[mask_source & mask_target].reset_index(drop=True)
    
    dropped = before_count - len(filtered)
    if dropped > 0:
        logger.warning(f"🗑️ Dropped {dropped} orphan relationship(s).")
        
    return filtered

# def filter_orphan_relationships(
#     relationships: pd.DataFrame,
#     entities: pd.DataFrame,
# ) -> pd.DataFrame:
#     """Remove relationships whose source or target has no entity entry.

#     After LLM graph extraction, the model may hallucinate entity
#     names in relationships that have no corresponding entity row.
#     This function drops those dangling references so downstream
#     processing never encounters broken graph edges.

#     Parameters
#     ----------
#     relationships:
#         Merged relationship DataFrame with at least ``source``
#         and ``target`` columns.
#     entities:
#         Merged entity DataFrame with at least a ``title`` column.

#     Returns
#     -------
#     pd.DataFrame
#         Relationships filtered to only those whose ``source``
#         and ``target`` both appear in ``entities["title"]``.
#     """
#     if relationships.empty or entities.empty:
#         return relationships.iloc[0:0].reset_index(drop=True)

#     entity_titles = set(entities["title"])
#     before_count = len(relationships)
#     mask = relationships["source"].isin(entity_titles) & relationships["target"].isin(
#         entity_titles
#     )
#     filtered = relationships[mask].reset_index(drop=True)
#     dropped = before_count - len(filtered)
#     if dropped > 0:
#         logger.warning(f"🗑️ Dropped {dropped} orphan relationship(s) referencing non-existent entities (Hallucinations).")
#     return filtered


def _empty_entities_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["title", "type", "description", "source_id"])


def _empty_relationships_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["source", "target", "weight", "description", "source_id"]
    )

