from app.indexing.operations.entity_resolution.identity_tracker import IdentityTracker
from app.indexing.operations.entity_resolution.llm_resolver import LLMResolver
from app.indexing.operations.entity_resolution.core_resolver import CoreResolver

from typing import Tuple
import pandas as pd


class EntityResolutionEngine:
    def __init__(self, core_resolver: CoreResolver, llm_resolver: LLMResolver):
        self.core = core_resolver
        self.llm = llm_resolver

    async def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        if df.empty:
            return df, {}

        tracker = IdentityTracker()
        
        # 1. RÉSOLUTION CORE (Phonétique + Encyclopédie)
        # Le nouveau CoreResolver renvoie (df, dict) -> Parfaitement compatible
        df, core_mappings = self.core.resolve(df)
        for old, new in core_mappings.items():
            tracker.add_mapping(old, new)

        # 2. RÉSOLUTION LLM (Anchoring + Pyramide)
        df, llm_mappings = await self.llm.resolve_complex_cases(df)
        for old, new in llm_mappings.items():
            tracker.add_mapping(old, new)

        # 3. GÉNÉRATION DU MAPPING FINAL
        final_map = tracker.resolve()

        # 4. HARMONISATION DU DATAFRAME
        # On applique le mapping sur les titres
        df["title"] = df["title"].replace(final_map)
        
        # Sécurité : Si une entité a un canonical_id (ID fixe Sira), 
        # son titre DOIT être cet ID pour garantir la cohérence du graphe.
        mask = df["canonical_id"].notna()
        df.loc[mask, "title"] = df.loc[mask, "canonical_id"]

        # 5. FUSION FINALE DES LIGNES DOUBLONNES
        # Maintenant que bcp d'entités ont le même titre, on les merge physiquement
        final_df = self._aggregate_entities(df)

        return final_df, final_map

    def _aggregate_entities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Regroupe les entités qui ont le même titre après résolution."""
        if df.empty:
            return df

        # On s'assure que description est une liste pour pouvoir les sommer
        if not isinstance(df.iloc[0]["description"], list):
            df["description"] = df["description"].apply(lambda d: [d] if isinstance(d, str) else d)

        agg_rules = {
            "description": "sum", # Fusionne les listes de strings
            "source_id": "sum",    # Fusionne les listes d'IDs sources
            "frequency": "sum",
            "canonical_id": "first"
        }
        
        # On garde 'type' dans le groupby pour ne pas fusionner un humain et un lieu 
        # qui auraient le même nom (homonymes)
        return df.groupby(["title", "type"], sort=False).agg(agg_rules).reset_index()