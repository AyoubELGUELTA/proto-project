import pandas as pd
import logging
from typing import List, Dict, Tuple, Optional, Any
from phonetics import dmetaphone
from Levenshtein import ratio

from app.core.config.graph_config import LEVENSHTEIN_SCORE_MERGE_TRIGGER

from app.core.data_model.entity import EntityModel
from app.core.data_model.encyclopedia import EncyclopediaEntry
from app.indexing.operations.entity_resolution.encyclopedia_manager import EncyclopediaManager

logger = logging.getLogger(__name__)

class CoreResolver:
    """
    Handles deterministic entity resolution through phonetic and string similarity algorithms.
    
    This resolver acts as a first-pass filter to merge lexical variants and anchor 
    extracted entities to the official Encyclopedia before any LLM-based resolution.
    """
   
    def __init__(self, encyclopedia: EncyclopediaManager, similarity_threshold: float = LEVENSHTEIN_SCORE_MERGE_TRIGGER):
        """
        Initializes the resolver.
        
        Args:
            encyclopedia: The manager for canonical reference data (SQL-backed).
            similarity_threshold: Levenshtein ratio (0.0 to 1.0) required to trigger a merge.
        """
        self.encyclopedia = encyclopedia
        self.similarity_threshold = similarity_threshold

    async def resolve(self, entities: List[EntityModel]) -> Tuple[List[EntityModel], Dict[str, str]]:        
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
        if not entities:
            return [], {}
        
        local_changes = {}
        
        # 1. Prepare Data via DataFrame for efficient clustering
        # We convert our Pydantic models to a DF for algorithmic processing
        df = pd.DataFrame([e.model_dump() for e in entities])
        
        # Phonetic Fingerprinting
        df["phonetic_key"] = df["title"].apply(lambda x: dmetaphone(str(x))[0] if x else "")
        logger.info(f"🔍 Phonetic fingerprinting completed for {len(df)} entities.")

        # 2. Iterative Algorithmic Merging
        merged_df = self._algorithmic_merging(df, local_changes)

        # 3. Encyclopedia Anchoring
        final_entities = []
        for _, row in merged_df.iterrows():
            # Conversion robuste : on s'assure que les listes sont des listes vides si None
            data = row.to_dict()

            # Cleanup NaN values to None, issued from the conversion
            data = {k: (v if pd.notna(v) else None) for k, v in data.items()}
            
            entity = EntityModel(**data)
            
            # Lookup in SQL Encyclopedia
            matches: List[EncyclopediaEntry] = await self.encyclopedia.find_match(
                entity.title, 
                entity.category
            )
            
            if len(matches) == 1:
                # Case A: Unique Match found (CORE_VALIDATED)

                canonical = matches[0]

                logger.info(f"✅ Encyclopedia Match: '{entity.title}' anchored to {canonical.id}")
                
                local_changes[entity.id] = canonical.id
                entity.canonical_id = canonical.id
                entity.review_status = "CORE_VALIDATED"
                
            elif len(matches) > 1:
                # Case B: Ambiguity - Flag for LLM intervention (PENDING)
                logger.warning(f"⚠️ Ambiguity: '{entity.title}' has {len(matches)} candidates.")
                # Important : on utilise le model_dump() pour que ce soit JSON-serializable
                entity.attributes["anchoring_candidates"] = [m.model_dump() for m in matches]
                entity.review_status = "PENDING" 
            
            else:
                # Case C: No match found
                entity.review_status = "NOT_KNOWN"

            final_entities.append(entity)

        return final_entities, local_changes

    def _algorithmic_merging(self, df: pd.DataFrame, changes: Dict[str, str]) -> pd.DataFrame:
        """
        Clusters entities based on frequency-first priority.
        
        The algorithm treats the most frequent name as the cluster 'pivot' 
        to ensure naming stability throughout the graph.
        """

        # Frequency-based sorting to ensure the dominant name survives as the cluster head        
        df = df.sort_values("frequency", ascending=False).reset_index(drop=True)
        
        merged_indices = set()
        final_rows = []

        for i, row in df.iterrows():
            if i in merged_indices: continue

            cluster = [row]
            merged_indices.add(i)

            for j, candidate in df.iloc[i+1:].iterrows():
                if j in merged_indices: continue
                
                if self._is_mergeable(row, candidate):
                    changes[candidate["id"]] = row["id"]
                    logger.debug(f"🔗 Merging variant '{candidate['title']}' into pivot '{row['title']}'")
                    cluster.append(candidate)
                    merged_indices.add(j)

            final_rows.append(self._aggregate_cluster(cluster))
            
        return pd.DataFrame(final_rows)

    def _is_mergeable(self, row: pd.Series, candidate: pd.Series) -> bool:
        """
        Deterministic check:
        1. Must share the same category (Human, Place, etc.)
        2. Must share phonetic sound AND string similarity.
        """
        # 1. Semantic Check (Zero overhead now)
        if row["category"] != candidate["category"]:
            return False

        # 2. Structural Check
        same_sound = row["phonetic_key"] == candidate["phonetic_key"]
        sim_ratio = ratio(str(row["slug"]), str(candidate["slug"]))
        
        return same_sound and sim_ratio >= self.similarity_threshold

    def _aggregate_cluster(self, cluster_rows: List[pd.Series]) -> Dict:
        """
        Collapses a group of similar entities into a single unified record.
        
        Merges source tracking IDs and joins descriptions with a pipe delimiter ("|") 
        to preserve context for the subsequent Summarization phase.
        """
               
        main = cluster_rows[0]        
        
        # 1. Traceability: Union of source_ids (deduplicated)
        all_sources = []
        for r in cluster_rows:
            all_sources.extend(r.get("source_ids", []))
        unique_sources = list(set(all_sources))
        
        # 2. Importance: Sum of frequencies and Max rank
        # If 'A' appears 3 times and 'A'' 2 times, the total is 5.
        total_frequency = sum(r.get("frequency", 1) for r in cluster_rows)

        # Keep the highest rank in the cluster to maintain entity significance
        max_rank = max(r.get("rank", 1) for r in cluster_rows)
        
        # 3. Context: Merge descriptions
        descriptions = set(filter(None, [str(r.get("description", "")) for r in cluster_rows]))
        
        # 4. Attributes & Communities: Merge dictionaries and lists
        merged_attributes = {}
        merged_communities = set()
        for r in cluster_rows:
            merged_attributes.update(r.get("attributes", {}))
            merged_communities.update(r.get("community_ids", []))

        # The returned dictionary must match the EntityModel fields
        return {
            "id": main["id"],
            "title": main["title"], # Will be re-normalized by the EntityModel validator
            "slug": main.get("slug"), 
            "type": main["type"],
            "category": main.get("category"), # Will be re-calculated by the model_validator if None
            "description": " | ".join(descriptions),
            "frequency": total_frequency,
            "source_ids": unique_sources,
            "rank": max_rank,
            "community_ids": list(merged_communities),
            "attributes": merged_attributes,
            "canonical_id": None,
            "review_status": "NOT_KNOWN"
        }