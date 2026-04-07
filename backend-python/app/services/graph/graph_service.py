import asyncio
import pandas as pd
import logging
from typing import List, Any, Tuple
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.indexing.operations.graph.utils import filter_orphan_relationships
from indexing.operations.text.text_utils import similarity, normalize_entity_name

from app.indexing.operations.graph.extract_graph import GraphExtractor
from app.services.llm.parser import LLMParser
from app.indexing.operations.entity_resolution.core_resolver import CoreResolver
from app.indexing.operations.entity_resolution.llm_resolver import LLMResolver
from app.core.config.graph_config import EntityResolvingConfig

logger = logging.getLogger(__name__)

class GraphService:
    def __init__(self, 
                 extractor: GraphExtractor, 
                 summarizer, 
                 parser: LLMParser, 
                 core_resolver: CoreResolver, 
                 llm_resolver: LLMResolver):
        self.extractor = extractor
        self.summarizer = summarizer
        self.parser = parser
        self.core_resolver = core_resolver 
        self.llm_resolver = llm_resolver

    async def run_pipeline(
        self, 
        text_units: List[Any], 
        domain_context: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        # 1. EXTRACTION & PARSING
        tasks = [self.extractor(u.text, domain_context) for u in text_units]
        raw_results = await asyncio.gather(*tasks)
        source_ids = [u.id for u in text_units]
        entities_df, relationships_df = self.parser.to_dataframes(raw_results, source_ids)

        if entities_df.empty:
            return entities_df, relationships_df

        # 2. RÉSOLUTION D'ENTITÉS (Pyramidale)
        entities_df, global_mapping = await self._perform_entity_resolution(entities_df)

        # 3. AGGRÉGATION & SUMMARIZATION DES ENTITÉS
        # On fusionne les descriptions brutes de toutes les entités résolues
        entities_df = await self._summarize_entity_descriptions(entities_df)
        entities_df["frequency"] = entities_df["source_id"].apply(len)

        # 4. MISE À JOUR DES RELATIONS (Mapping + Dédoublonnage)
        if not relationships_df.empty:
            # Application du mapping (ex: C -> B devient A -> B)
            if global_mapping:
                relationships_df["source"] = relationships_df["source"].replace(global_mapping)
                relationships_df["target"] = relationships_df["target"].replace(global_mapping)
            
            # Nettoyage des Self-loops (A -> A) suite au mapping
            relationships_df = relationships_df[relationships_df["source"] != relationships_df["target"]]

            # Second GroupBy : Fusionne les relations devenues identiques (A->B et A->B)
            relationships_df = (
                relationships_df.groupby(["source", "target"], sort=False)
                .agg({
                    "description": "sum", # On combine les listes de descriptions
                    "weight": "sum",      # On additionne les forces de relation
                    "source_id": "sum"    # On combine les sources
                })
                .reset_index()
            )

        # 5. FILTRAGE FINAL DES ORPHELINS
        # On le fait à la fin pour être sûr de ne pas supprimer une relation 
        # dont le "target" a été renommé par le Resolver
        relationships_df = filter_orphan_relationships(
            relationships=relationships_df,
            entities=entities_df
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
        MAX_BATCH = EntityResolvingConfig.max_cluster_batch 
        if len(cluster_df) <= MAX_BATCH:
            return await self.llm_resolver.resolve_cluster(cluster_df, entity_type)

        # Map : Parallel batches
        chunks = [cluster_df.iloc[i:i + MAX_BATCH] for i in range(0, len(cluster_df), MAX_BATCH)]
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
        """Regroupe les lignes par titre et concatène les listes sans résumer."""
        return df.groupby(["title", "type"], sort=False).agg({
            "description": "sum", # Concaténation de listes [desc1] + [desc2]
            "source_id": "sum",
            "frequency": "sum", # Si déjà calculé, sinon on le fera après
            "canonical_id": "first"
        }).reset_index()

    async def _summarize_entity_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Appelle le LLM pour créer un résumé unique à partir de la liste de descriptions.
        C'est l'étape la plus coûteuse, placée après toutes les fusions.
        """
        async def summarize_row(row):
            # On nettoie les doublons exacts dans les descriptions collectées
            descs = list(set(row["description"])) 
            if len(descs) < 3:
                return " ".join(descs) if descs else ""
            
            return await self.summarizer(entity_name=row["title"], descriptions=descs)

        # On crée des tâches pour résumer chaque entité unique
        tasks = [summarize_row(row) for _, row in df.iterrows()]
        df["description"] = await asyncio.gather(*tasks)
        return df