import pandas as pd
from typing import List, Tuple, Dict, Any
from .graph_extractor import GraphExtractor
from .utils import filter_orphan_relationships


async def extract_graph(
    text_units: List[Dict[str, Any]], 
    extractor: GraphExtractor,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orchestre l'extraction : Tuples -> Dicts -> DataFrames -> Merged Graph.
    """
    raw_entities = []
    raw_relationships = []


    for unit in text_units:
        # 1. Extraction (On passe metadata comme context)
        tuples = await extractor(text=unit["text"], context=unit.get("metadata", ""))
        
        # 2. Parsing des tuples et injection du source_id
        for t in tuples:
            tag = t[0].lower()
            if "entity" in tag and len(t) >= 4:
                raw_entities.append({
                    "title": t[1].upper(), "type": t[2].upper(), 
                    "description": t[3], "source_id": unit["id"]
                })
            elif "relationship" in tag and len(t) >= 5:
                raw_relationships.append({
                    "source": t[1].upper(), "target": t[2].upper(), 
                    "description": t[3], "weight": float(t[4]) if str(t[4]).replace('.','').isdigit() else 1.0,
                    "source_id": unit["id"]
                })

    if not raw_entities:
        return pd.DataFrame(), pd.DataFrame()

    # 3. Transformation en DataFrame + Grouping (Scalable)
    entities_df = pd.DataFrame(raw_entities)

    print(f"DEBUG: raw_entities source_id type: {type(raw_entities[0]['source_id'])}")

    entities = (
        entities_df.groupby(["title", "type"], sort=False)
        .agg({"description": list, "source_id": list})
        .reset_index()
        )

    print(f"DEBUG: entities source_id content: {entities['source_id'].iloc[0]}")

    entities["frequency"] = entities["source_id"].apply(len)

    relationships_df = pd.DataFrame(raw_relationships)
    relationships = (
        relationships_df.groupby(["source", "target"], sort=False)
        .agg({"description": list, "weight": "sum", "source_id": list})
        .reset_index()
    )

    # 4. Nettoyage final
    relationships = filter_orphan_relationships(relationships, entities)

    return entities, relationships