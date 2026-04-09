# Copyright (C) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Utility functions for graph extraction operations."""


import pandas as pd


def filter_orphan_relationships(
    relationships: pd.DataFrame,
    entities: pd.DataFrame,
) -> pd.DataFrame:
    """Remove relationships whose source or target has no entity entry.

    After LLM graph extraction, the model may hallucinate entity
    names in relationships that have no corresponding entity row.
    This function drops those dangling references so downstream
    processing never encounters broken graph edges.

    Parameters
    ----------
    relationships:
        Merged relationship DataFrame with at least ``source``
        and ``target`` columns.
    entities:
        Merged entity DataFrame with at least a ``title`` column.

    Returns
    -------
    pd.DataFrame
        Relationships filtered to only those whose ``source``
        and ``target`` both appear in ``entities["title"]``.
    """
    if relationships.empty or entities.empty:
        return relationships.iloc[0:0].reset_index(drop=True)

    entity_titles = set(entities["title"])
    before_count = len(relationships)
    mask = relationships["source"].isin(entity_titles) & relationships["target"].isin(
        entity_titles
    )
    filtered = relationships[mask].reset_index(drop=True)
    dropped = before_count - len(filtered)
    if dropped > 0:
        print(f"Dropped {dropped} relationship(s) referencing non-existent entities.")
    return filtered


def _empty_entities_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["title", "type", "description", "source_id"])


def _empty_relationships_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["source", "target", "weight", "description", "source_id"]
    )

