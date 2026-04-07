# Logique de matching (Deterministic + Phonetic)
import pandas as pd
from typing import List, Dict, Tuple
import logging
from phonetics import dmetaphone
from Levenshtein import ratio
from app.indexing.operations.entity_resolution.encyclopedia_manager import EncyclopediaManager

logger = logging.getLogger(__name__)

class CoreResolver:
    def __init__(self, encyclopedia: EncyclopediaManager, similarity_threshold: float = 0.85):
        self.encyclopedia = encyclopedia
        self.similarity_threshold = similarity_threshold

    def resolve(self, entities_df: pd.DataFrame) -> pd.DataFrame:
        if entities_df.empty: return entities_df
        
        df = entities_df.copy()

        # ÉTAPE 1 : On donne une chance à TOUT LE MONDE d'être groupé phonétiquement
        # On ne cherche pas encore dans l'encyclopédie
        df["phonetic_key"] = df["title"].apply(lambda x: dmetaphone(x)[0])

        # ÉTAPE 2 : On crée nos clusters (Umar et Oumar se retrouvent ici dans le même sac)
        resolved_df = self._merge_phonetic_groups(df)

        # ÉTAPE 3 : ANCRAGE - On vérifie chaque cluster par rapport à l'Encyclopédie
        resolved_df["canonical_id"] = None
        for idx, row in resolved_df.iterrows():
            # On check le titre du cluster (le plus fréquent)
            match = self.encyclopedia.find_match(row["title"], row["type"])
            if match:
                resolved_df.at[idx, "canonical_id"] = match["ID"]
                resolved_df.at[idx, "title"] = match["CANONICAL_NAME"]
            else:
                # OPTIONNEL : On pourrait aussi checker si l'un des ALIASES 
                # est présent dans les titres originaux du cluster si on voulait être ultra-fin.
                pass
        
        return resolved_df

    def _merge_phonetic_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fusionne les entités si :
        1. Elles ont le même TYPE.
        2. Elles ont la même PHONETIC_KEY.
        3. Le LEVENSHTEIN entre leurs titres est > threshold.
        """
        # On trie par fréquence pour garder le titre le plus commun comme "chef de file"
        df = df.sort_values("frequency", ascending=False)
        df = df.reset_index(drop=True)

        merged_indices = set()
        final_rows = []

        for i, row in df.iterrows():
            if i in merged_indices:
                continue

            # On cherche des partenaires de fusion parmi les suivants
            current_cluster = [row]
            merged_indices.add(i)

            for j, candidate in df.iloc[i+1:].iterrows():
                if j in merged_indices:
                    continue
                
                # Condition de fusion : Même type + Même son + Écriture proche
                same_type = row["type"] == candidate["type"]
                same_sound = row["phonetic_key"] == candidate["phonetic_key"]
                similar_writing = ratio(row["title"], candidate["title"]) >= self.similarity_threshold

                if same_type and same_sound and similar_writing:
                    current_cluster.append(candidate)
                    merged_indices.add(j)

            # Agrégation du cluster
            final_rows.append(self._aggregate_cluster(current_cluster))

        return pd.DataFrame(final_rows)

    def _aggregate_cluster(self, cluster_rows: List[pd.Series]) -> Dict:
        """ Fusionne une liste de lignes (Series) en une seule entité. """
        main = cluster_rows[0]
        return {
            "title": main["title"],
            "type": main["type"],
            "description": [d for r in cluster_rows for d in r["description"]],
            "source_id": [s for r in cluster_rows for s in r["source_id"]],
            "frequency": sum(r["frequency"] for r in cluster_rows),
            "canonical_id": main["canonical_id"]
        }