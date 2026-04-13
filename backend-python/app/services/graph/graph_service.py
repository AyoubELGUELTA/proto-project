import asyncio
import pandas as pd
import logging
from typing import List, Any, Tuple

from app.indexing.operations.graph.utils import filter_orphan_relationships
from app.indexing.operations.graph.graph_extractor import EntityAndRelationExtractor
from app.indexing.operations.graph.summarize_manager import SummarizeManager

from app.services.llm.parser import LLMParser
from app.indexing.operations.entity_resolution.resolution_engine import EntityResolutionEngine


class GraphService:
    def __init__(
        self, 
        extractor: EntityAndRelationExtractor, 
        summarizer: SummarizeManager, 
        parser: LLMParser, 
        resolution_engine: EntityResolutionEngine
    ):
        self.extractor = extractor
        self.summarizer = summarizer
        self.parser = parser
        self.resolution_engine = resolution_engine

    async def run_pipeline(
        self, 
        text_units: List[Any], 
        domain_context: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        # 1. EXTRACTION & PARSING
        # On garde l'extraction asynchrone par batch
        tasks = [self.extractor(u.text, domain_context) for u in text_units]
        raw_results = await asyncio.gather(*tasks)
        source_ids = [u.id for u in text_units]
        
        entities_df, relationships_df = self.parser.to_dataframes(raw_results, source_ids)

        if entities_df.empty:
            return entities_df, relationships_df

        # 2. RÉSOLUTION D'ENTITÉS (La magie opère ici)
        # L'Engine s'occupe de tout : Core, LLM, Mapping transitif et Fusion des lignes
        entities_df, global_mapping = await self.resolution_engine.run(entities_df)
        
        # Mise à jour de la fréquence post-fusion
        entities_df["frequency"] = entities_df["source_id"].apply(len)

        # 3. MISE À JOUR DES RELATIONS
        if not relationships_df.empty:
            relationships_df = self._process_relationships(relationships_df, global_mapping)

        # 4. FILTRAGE DES ORPHELINS
        relationships_df = filter_orphan_relationships(relationships_df, entities_df)

        # 5. SUMMARIZATION
        # On délègue à la fonction spécialisée
        entities_df, relationships_df = await self.summarizer.summarize_all(entities_df=entities_df,
            relationships_df=relationships_df)

        return entities_df, relationships_df

    def _process_relationships(self, df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
        """Logique isolée pour nettoyer et agréger les relations."""
        # Application du mapping global (A -> C)
        if mapping:
            df["source"] = df["source"].replace(mapping)
            df["target"] = df["target"].replace(mapping)
        
        # Nettoyage des boucles infinies (A -> A)
        df = df[df["source"] != df["target"]]
        
        # Agrégation des doublons de relations (ex: 2 fois la même relation dans 2 chapitres)
        return (
            df.groupby(["source", "target"], sort=False)
            .agg({
                "description": list, 
                "weight": "sum", 
                "source_id": lambda x: list(set([
                    item for sublist in (x if isinstance(x.iloc[0], list) else [x]) 
                    for item in sublist
                ])) 
            })
            .reset_index()
        )