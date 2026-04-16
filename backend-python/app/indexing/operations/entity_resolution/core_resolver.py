import pandas as pd
from typing import List, Dict, Tuple
import logging
from phonetics import dmetaphone
from Levenshtein import ratio
from app.indexing.operations.entity_resolution.encyclopedia_manager import EncyclopediaManager
from app.models.domain import SiraEntityType

logger = logging.getLogger(__name__)

class CoreResolver:
    """
    Handles deterministic entity resolution through phonetic and string similarity algorithms.
    
    The CoreResolver reduces entity duplication before LLM intervention by:
    1. Grouping lexical variants (phonetic + Levenshtein distance).
    2. Anchoring clusters to the Encyclopedia (Canonical Source).
    3. Aggregating metadata (sources, descriptions, frequencies) for merged entities.
    """
    def __init__(self, encyclopedia: EncyclopediaManager, similarity_threshold: float = 0.85):
        """
        Initializes the resolver with a reference encyclopedia.
        
        Args:
            encyclopedia: Master reference for canonical entity validation.
            similarity_threshold: Levenshtein ratio (0.0 to 1.0) required to trigger a merge.
        """
        self.encyclopedia = encyclopedia
        self.similarity_threshold = similarity_threshold

    def resolve(self, entities_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Executes the three-stage deterministic resolution pipeline.
        
        Steps:
        1. Phonetic Key Generation: Using Double Metaphone for sound-based indexing.
        2. Algorithmic Merging: Clustering variants within the current DataFrame.
        3. Encyclopedia Anchoring: Attempting to map clusters to known canonical IDs.
        
        Returns:
            - resolved_df: Cleaned DataFrame with merged rows and canonical_ids.
            - local_changes: A mapping dictionary {old_name: new_canonical_name_or_id}.
        """
        if entities_df.empty:
            return entities_df, {}
        
        local_changes = {}
        df = entities_df.copy()
        
        if "canonical_id" not in df.columns:
            df["canonical_id"] = None

        # 1. Phonetic Fingerprinting
        # Allows matching 'Muhammad' and 'Muhamad' even with spelling variations
        df["phonetic_key"] = df["title"].apply(lambda x: dmetaphone(str(x))[0] if x else "")#TODO put aswell the second proposition of dmetaphone later
        logger.info(f"🔍 Phonetic fingerprinting completed for {len(df)} entities.")

        # 2. Iterative Algorithmic Merging
        resolved_df = self._algoritmic_merging(df, local_changes)

        # 3. Reference Data Anchoring (Encyclopedia)
        for idx, row in resolved_df.iterrows():
            matches = self.encyclopedia.find_match(row["title"], row["type"])
            
            # Case A: Unique Match found - Direct Anchoring
            if len(matches) == 1:
                old_title = row["title"]
                new_title = matches[0]["CANONICAL_NAME"]
                new_id = matches[0]["ID"]

                logger.info(f"✅ Encyclopedia Match: '{old_title}' anchored to ID {new_id}")

                local_changes[old_title] = new_id
                resolved_df.at[idx, "canonical_id"] = new_id
                resolved_df.at[idx, "title"] = new_title
            
            # Case B: Ambiguous Matches - Flagging for LLM intervention
            elif len(matches) > 1:
                logger.warning(f"⚠️ Ambiguity: '{row['title']}' has {len(matches)} candidates. Flagged for LLM.")
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
        """
        Clusters entities based on frequency-first priority.
        
        The algorithm treats the most frequent name as the cluster 'pivot' 
        to ensure naming stability throughout the graph.
        """

        # Frequency-based sorting to ensure the dominant name survives as the cluster head        
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
                    # Record the redirection for relationship remapping
                    changes[candidate["title"]] = row["title"]
                    logger.debug(f"🔗 Merging variant '{candidate['title']}' into pivot '{row['title']}'")
                    current_cluster.append(candidate)
                    merged_indices.add(j)

            final_rows.append(self._aggregate_cluster(current_cluster))
        logger.info(f"📉 Algorithmic merging reduced DF from {len(df)} to {len(final_rows)} entities.")
        return pd.DataFrame(final_rows)

    def _is_mergeable(self, row: pd.Series, candidate: pd.Series) -> bool:
        """
        Hybrid comparison logic combining sound and edit distance.
        
        Validation layers:
        1. Structural: Both phonetic keys must match AND Levenshtein ratio > threshold.
        2. Semantic: Both entities must belong to the same Category (SiraEntityType).
        """

        # 1. Forme
        same_sound = row["phonetic_key"] == candidate["phonetic_key"]#TODO handles the second dmetaphone phonetic proposition
        sim_ratio = ratio(str(row["title"]), str(candidate["title"]))
        
        if not (same_sound and sim_ratio >= self.similarity_threshold):
            return False

        # 2. Fond (Types)
        cat_a = SiraEntityType.get_category(row["type"])
        cat_b = SiraEntityType.get_category(candidate["type"])

        return cat_a == cat_b

    def _aggregate_cluster(self, cluster_rows: List[pd.Series]) -> Dict:
        """
        Collapses a group of similar entities into a single unified record.
        
        Merges source tracking IDs and joins descriptions with a pipe delimiter ("|") 
        to preserve context for the subsequent Summarization phase.
        """
       
        main = cluster_rows[0]
        
        # Aggregate unique source IDs across the cluster
        all_sources = []
        for r in cluster_rows:
            sid = r["source_id"]
            if isinstance(sid, list): all_sources.extend(sid)
            else: all_sources.append(sid)

        # Collect unique descriptions for later LLM summarization
        descriptions = set(filter(None, [str(r["description"]) for r in cluster_rows]))

        return {
            "title": main["title"],
            "type": main["type"],
            "description": " | ".join(descriptions),
            "source_id": list(set(all_sources)),
            "frequency": sum(r.get("frequency", 1) for r in cluster_rows),
            "canonical_id": main.get("canonical_id")
        }