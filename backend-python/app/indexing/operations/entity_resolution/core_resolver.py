# Logique de matching (Deterministic + Phonetic)
import pandas as pd
from typing import List, Dict, Tuple
import logging
from phonetics import dmetaphone
from Levenshtein import ratio
from app.indexing.operations.entity_resolution.encyclopedia_manager import EncyclopediaManager
from app.models.domain import SiraEntityType

logger = logging.getLogger(__name__)

class CoreResolver:
    def __init__(self, encyclopedia: EncyclopediaManager, similarity_threshold: float = 0.85):
        self.encyclopedia = encyclopedia
        self.similarity_threshold = similarity_threshold

    def resolve(self, entities_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Exécute la résolution déterministe.
        Retourne : (Le DataFrame fusionné, Le dictionnaire des changements effectués)
        """
        if entities_df.empty:
            return entities_df, {}
        
        # On garde une trace des noms originaux avant toute modif
        local_changes = {}
        
        df = entities_df.copy()
        if "canonical_id" not in df.columns:
            df["canonical_id"] = None

        # 1. PRÉPARATION PHONÉTIQUE
        df["phonetic_key"] = df["title"].apply(lambda x: dmetaphone(str(x))[0] if x else "")

        # 2. MERGING ALGORITHMIQUE (Phonétique + Levenshtein)
        # On passe le dictionnaire local_changes pour le remplir pendant la fusion
        resolved_df = self._algoritmic_merging(df, local_changes)

        # 3. ANCRAGE ENCYCLOPÉDIE
        for idx, row in resolved_df.iterrows():
            matches = self.encyclopedia.find_match(row["title"], row["type"])
            
            if len(matches) == 1:
                old_title = row["title"]
                new_title = matches[0]["CANONICAL_NAME"]
                new_id = matches[0]["ID"]
                
                # On enregistre le changement : Nom de cluster -> ID Encyclopédie
                local_changes[old_title] = new_id
                
                resolved_df.at[idx, "canonical_id"] = new_id
                resolved_df.at[idx, "title"] = new_title
            
            elif len(matches) > 1:
                # On prépare les candidats pour le LLMResolver (étape suivante)
                resolved_df.at[idx, "anchoring_candidates"] = [
                    {
                        "ID": m["ID"],
                        "CANONICAL_NAME": m["CANONICAL_NAME"],
                        "CORE_SUMMARY": m.get("CORE_SUMMARY", ""),
                        "TYPE": m["TYPE"] 
                    }
                    for m in matches
                ]

        return resolved_df, local_changes

    def _algoritmic_merging(self, df: pd.DataFrame, changes: dict) -> pd.DataFrame:
        """Fusionne les variantes orthographiques et remplit le mapping."""
        # On trie par fréquence pour que le nom le plus commun devienne le 'parent'
        df['frequency'] = df.groupby('title')['title'].transform('count')
        df = df.sort_values("frequency", ascending=False).reset_index(drop=True)
        
        merged_indices = set()
        final_rows = []

        for i, row in df.iterrows():
            if i in merged_indices: continue

            current_cluster = [row]
            merged_indices.add(i)

            for j, candidate in df.iloc[i+1:].iterrows():
                if j in merged_indices: continue
                
                if self._is_mergeable(row, candidate):
                    # On a trouvé un variant ! On enregistre le mapping
                    changes[candidate["title"]] = row["title"]
                    current_cluster.append(candidate)
                    merged_indices.add(j)

            final_rows.append(self._aggregate_cluster(current_cluster))

        return pd.DataFrame(final_rows)

    def _is_mergeable(self, row: pd.Series, candidate: pd.Series) -> bool:
        """Logique centrale de comparaison."""
        # 1. Forme
        same_sound = row["phonetic_key"] == candidate["phonetic_key"]
        sim_ratio = ratio(str(row["title"]), str(candidate["title"]))
        
        if not (same_sound and sim_ratio >= self.similarity_threshold):
            return False

        # 2. Fond (Types)
        cat_a = SiraEntityType.get_category(row["type"])
        cat_b = SiraEntityType.get_category(candidate["type"])

        return cat_a == cat_b

    def _aggregate_cluster(self, cluster_rows: List[pd.Series]) -> Dict:
        """Fusionne physiquement les lignes du cluster."""
        main = cluster_rows[0]
        
        # Agrégation intelligente des source_id
        all_sources = []
        for r in cluster_rows:
            sid = r["source_id"]
            if isinstance(sid, list): all_sources.extend(sid)
            else: all_sources.append(sid)

        # Agrégation des descriptions
        descriptions = set(filter(None, [str(r["description"]) for r in cluster_rows]))

        return {
            "title": main["title"],
            "type": main["type"],
            "description": " | ".join(descriptions),
            "source_id": list(set(all_sources)),
            "frequency": sum(r.get("frequency", 1) for r in cluster_rows),
            "canonical_id": main.get("canonical_id")
        }