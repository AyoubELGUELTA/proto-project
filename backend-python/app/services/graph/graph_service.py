import asyncio
import pandas as pd
import logging
from typing import List, Any, Tuple
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.indexing.operations.graph.utils import filter_orphan_relationships
from app.indexing.operations.text.text_utils import similarity, normalize_entity_name

from app.indexing.operations.graph.extract_graph import GraphExtractor
from app.indexing.operations.graph.summarize_extractor import SummarizeExtractor
from app.indexing.operations.graph.summarize_descriptions import summarize_descriptions

from app.services.llm.parser import LLMParser
from app.services.llm.service import LLMService
from app.indexing.operations.entity_resolution.core_resolver import CoreResolver
from app.indexing.operations.entity_resolution.llm_resolver import LLMResolver
from app.core.config.graph_config import ENTITY_BATCH_SIZE, MAX_CLUSTER_BATCH

logger = logging.getLogger(__name__)

class GraphService:
    def __init__(self, 
                 extractor: GraphExtractor, 
                 summarize_extractor: SummarizeExtractor, 
                 parser: LLMParser, 
                 core_resolver: CoreResolver, 
                 llm_resolver: LLMResolver):
        self.extractor = extractor
        self.summarize_extractor = summarize_extractor
        self.parser = parser
        self.core_resolver = core_resolver 
        self.llm_resolver = llm_resolver

    async def run_pipeline(
        self, 
        text_units: List[Any], 
        domain_context: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        # 1. Extraction 
        tasks = [self.extractor(u.text, domain_context) for u in text_units]
        raw_results = await asyncio.gather(*tasks)
        
        source_ids = [u.id for u in text_units]
        
        entities_df, relationships_df = self.parser.to_dataframes(raw_results, source_ids)
        
        print(f"\n✅ DEBUG LIGNE 16 EXTRACT_GRAPH: IDS = {source_ids}")
        print(f"📊 [STAGE 1] Post-Parsing: Entities {entities_df.shape}, Relations {relationships_df.shape}")
        if not entities_df.empty:
            print(f"🔍 IDs uniques trouvés: {entities_df['source_id'].unique()}")

        if entities_df.empty:
            return entities_df, relationships_df

        # 2. RÉSOLUTION D'ENTITÉS (Pyramidale)
        entities_df, global_mapping = await self._perform_entity_resolution(entities_df)
        entities_df["frequency"] = entities_df["source_id"].apply(len)

        print(f"📊 [STAGE 2] Post-Resolution: {len(entities_df)} unique entities")

        # 3. MISE À JOUR DES RELATIONS
        if not relationships_df.empty:
            if global_mapping:
                relationships_df["source"] = relationships_df["source"].replace(global_mapping)
                relationships_df["target"] = relationships_df["target"].replace(global_mapping)
            
            relationships_df = relationships_df[relationships_df["source"] != relationships_df["target"]]
            # --- LOG AVANT AGGREGATION ---
            print(f"🔍 Pre-agg sample source_id: {relationships_df['source_id'].iloc[0]}")
            
            relationships_df = (
                relationships_df.groupby(["source", "target"], sort=False)
                .agg({
                    "description": list, 
                    "weight": "sum", 
                    "source_id": lambda x: list(set(x)) if isinstance(x.iloc[0], str) else list(set([item for sublist in x for item in sublist]))
                })
                .reset_index()
            )
            print(f"📊 [STAGE 3] Post-Aggregation: {len(relationships_df)} relations")
            print(f"🔍 Post-agg sample source_id: {relationships_df['source_id'].iloc[0]}")

        # 4. FILTRAGE DES ORPHELINS
        relationships_df = filter_orphan_relationships(relationships_df, entities_df)

        # 5. AGGRÉGATION & SUMMARIZATION (Entités + Relations) via l'opérateur externe !
        print("🧠 Starting Summarization (Descriptions merging)...")
        entities_df, relationships_df = await summarize_descriptions(
            entities_df=entities_df,
            relationships_df=relationships_df,
            extractor=self.summarize_extractor,
            num_threads=ENTITY_BATCH_SIZE 
        )

        return entities_df, relationships_df

    async def _perform_entity_resolution(self, entities_df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """Orchestration de la résolution : Core -> Hybride -> Pyramide."""
        # Core resolution (Encyclopédie)
        df = self.core_resolver.resolve(entities_df)

        orphans = df[df["canonical_id"].isna()].copy()
        anchored = df[~df["canonical_id"].isna()].copy()

        if orphans.empty:
            return df, {}

        global_mapping = {}
        for entity_type, group in orphans.groupby("type"):
            sub_clusters = self._create_hybrid_clusters(group)
            for cluster in sub_clusters:
                mapping = await self._resolve_pyramid(cluster, str(entity_type))
                global_mapping.update(mapping)

        global_mapping = self._apply_transitive_mapping(global_mapping)
        
        # On applique le mapping sur les titres des orphelins
        orphans["title"] = orphans["title"].replace(global_mapping)

        # Fusion des lignes (sans summarization encore, juste regroupement des listes de descriptions)
        final_df = pd.concat([anchored, orphans], ignore_index=True)
        final_df = self._final_merge(final_df)

        return final_df, global_mapping

    async def _resolve_pyramid(self, cluster_df: pd.DataFrame, entity_type: str) -> dict:
        """Résolution récursive par batch pour ne perdre aucun lien sémantique."""
        if len(cluster_df) <= MAX_CLUSTER_BATCH:
            return await self.llm_resolver.resolve_cluster(cluster_df, entity_type)

        # Map : Parallel batches
        chunks = [cluster_df.iloc[i:i + MAX_CLUSTER_BATCH] for i in range(0, len(cluster_df), MAX_CLUSTER_BATCH)]
        tasks = [self.llm_resolver.resolve_cluster(chunk, entity_type) for chunk in chunks]
        mappings = await asyncio.gather(*tasks)
        
        combined_map = {}
        for m in mappings: combined_map.update(m)
        combined_map = self._apply_transitive_mapping(combined_map)

        # Reduce : On fait remonter les survivants
        temp_df = cluster_df.copy()
        temp_df["title"] = temp_df["title"].replace(combined_map)
        survivors_df = self._final_merge(temp_df) 

        if len(survivors_df) < len(cluster_df):
            recursive_map = await self._resolve_pyramid(survivors_df, entity_type)
            combined_map.update(recursive_map)
        
        return self._apply_transitive_mapping(combined_map)

    def _apply_transitive_mapping(self, mapping: dict) -> dict:
        """A -> B, B -> C  => A -> C"""
        final_map = {}
        for key in mapping:
            path = set([key])
            current = mapping[key]
            while current in mapping and current not in path:
                path.add(current)
                current = mapping[current]
            final_map[key] = current
        return final_map

    def _create_hybrid_clusters(self, df: pd.DataFrame) -> list:
        """Clustering TF-IDF + Levenshtein + Contenance."""
        if len(df) <= 1: return [df]

        df = df.copy()
        df["norm_name"] = df["title"].apply(normalize_entity_name)
        
        texts = df.apply(lambda x: f"{x['title']} {' '.join(x['description'])}", axis=1).tolist()
        vectorizer = TfidfVectorizer(stop_words='english') 
        tfidf_matrix = vectorizer.fit_transform(texts)
        cosine_sim = cosine_similarity(tfidf_matrix)

        G = nx.Graph()
        G.add_nodes_from(range(len(df)))

        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                # Critères de proximité
                is_semantic = cosine_sim[i][j] >= 0.3
                name_i, name_j = df.iloc[i]["norm_name"], df.iloc[j]["norm_name"]
                is_contained = (name_i in name_j) or (name_j in name_i) if (len(name_i) > 2 and len(name_j) > 2) else False
                is_fuzzy = similarity(name_i, name_j) >= 0.7

                if is_semantic or is_contained or is_fuzzy:
                    G.add_edge(i, j)

        return [df.iloc[list(c)] for c in nx.connected_components(G)]

    def _final_merge(self, df: pd.DataFrame) -> pd.DataFrame:
        # On s'assure que description est une liste avant de faire sum
        if not df.empty and not isinstance(df.iloc[0]["description"], list):
            df["description"] = df["description"].apply(lambda d: [d] if isinstance(d, str) else d) # En gros on wrapp notre description dans une liste si c'est juste un str

        return df.groupby(["title", "type"], sort=False).agg({
            "description": "sum", 
            "source_id": "sum",
            "frequency": "sum",
            "canonical_id": "first"
        }).reset_index()
