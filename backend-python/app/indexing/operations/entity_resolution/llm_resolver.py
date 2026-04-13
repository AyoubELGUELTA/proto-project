import pandas as pd
import asyncio
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, Tuple
from app.services.llm.service import LLMService
from app.core.prompts.graph_prompts import (
    ENTITY_RESOLUTION_SYSTEM_PROMPT, 
    ENTITY_RESOLUTION_USER_PROMPT,
    ANCHORING_RESOLUTION_SYSTEM_PROMPT,
    ANCHORING_RESOLUTION_USER_PROMPT
)
from app.core.config.graph_config import MAX_CLUSTER_BATCH
from app.indexing.operations.text.text_utils import similarity, normalize_entity_name


class LLMResolver:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service


    async def resolve_complex_cases(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Elle coordonne l'Anchoring ET le Clustering Pyramidal.
        """
        all_llm_mappings = {}
        
        # 1. Gérer l'Anchoring (ceux qui ont des doutes avec l'encyclopédie)
        if "anchoring_candidates" in df.columns:
            ambiguous_df = df[df["anchoring_candidates"].notna()]
            if not ambiguous_df.empty:
                tasks = [self.resolve_anchoring(row) for _, row in ambiguous_df.iterrows()]
                results = await asyncio.gather(*tasks)
                
                for idx, result in zip(ambiguous_df.index, results):
                    choice = result.get("choice")
                    if choice and choice != "NEW_ENTITY":
                        old_name = df.at[idx, "title"]
                        all_llm_mappings[old_name] = choice
                        df.at[idx, "canonical_id"] = choice # On marque l'ID trouvé

        # 2. Gérer le Clustering Pyramidal (ceux qui n'ont pas d'ID mais se ressemblent)

        orphans = df[df["canonical_id"].isna()].copy()
        if not orphans.empty:
            # On groupe par type (Humain avec Humain, etc.)
            for entity_type, type_group in orphans.groupby("type"):
                # On crée des clusters sémantiques (TF-IDF / Levenshtein)
                clusters = self._create_hybrid_clusters(type_group) 
                
                for cluster in clusters:
                    # SI LE CLUSTER EST TROP GROS, ON LE COUPE EN BATCHS
                    if len(cluster) > MAX_CLUSTER_BATCH:
                        # On traite par morceaux
                        for i in range(0, len(cluster), MAX_CLUSTER_BATCH):
                            batch = cluster.iloc[i:i + MAX_CLUSTER_BATCH]
                            mapping = await self.resolve_cluster(batch, str(entity_type))
                            all_llm_mappings.update(mapping)
                    else:
                        # Cluster de taille normale
                        mapping = await self.resolve_cluster(cluster, str(entity_type))
                        all_llm_mappings.update(mapping)

        return df, all_llm_mappings

    async def resolve_cluster(self, cluster_df: pd.DataFrame, entity_type: str) -> Dict[str, str]:
        """
        Analyse un cluster de doublons potentiels via le LLM.
        Retourne un dictionnaire de mapping : {"NOM_A_CHANGER": "NOM_CANONIQUE"}
        """
        if len(cluster_df) < 2:
            return {}

        # 1. Préparation des candidats (On limite la description pour économiser les tokens)
        candidates_list = []
        for row in cluster_df.itertuples():
            # 'row.description' est une liste de strings [desc1, desc2...]
            # On les fusionne pour avoir tout le contexte accumulé avant de couper
            full_context = " ".join(set(row.description)) if isinstance(row.description, list) else str(row.description)
            
            # On nettoie les sauts de ligne pour garder le prompt compact
            clean_context = full_context.replace("\n", " ").strip()
            
            # On prend les 300 premiers caractères : assez pour le Nasab et les titres (assez pour toute la description normalement)
            snippet = clean_context[:250] + "..." if len(clean_context) > 300 else clean_context
            
            candidates_list.append(f"- {row.title} (Type: {entity_type}): {snippet}")

        candidates_text = "\n".join(candidates_list)

        # 2. Utilisation du LLMService (Workflow standardisé)
        try:
            tuples = await self.llm_service.ask_tuples(
                system_prompt=ENTITY_RESOLUTION_SYSTEM_PROMPT,
                user_prompt=ENTITY_RESOLUTION_USER_PROMPT.format(
                    entity_type=entity_type,
                    candidates=candidates_text
                )
            )

            mapping = {}
            if not tuples:
                print(f"Aucun merge identifié pour le cluster {entity_type}")
                return {}

            for t in tuples:
                # Format attendu : ["MERGE", "Original", "Target"]
                if len(t) >= 3 and t[0].upper() == "MERGE":
                    source, target = t[1].strip(), t[2].strip()
                    if source != target:
                        mapping[source] = target
            
            if mapping:
                print(f"🤝 LLM Resolved {len(mapping)} merges for {entity_type}")
            
            return mapping

        except Exception as e:
            print(f"❌ Erreur LLMResolver sur cluster {entity_type}: {e}")
            return {}
        

    async def resolve_anchoring(self, row: pd.Series) -> Dict[str, Any]:
        """
        Cas spécifique : L'entité a plusieurs suspects dans l'encyclopédie.
        Le LLM doit choisir le bon ID ou déclarer 'NEW_ENTITY'.
        """
        candidates = row.get("anchoring_candidates", [])
        if not candidates:
            return {"choice": "NEW_ENTITY"}

        # 1. Formatage des données
        candidates_text = self._format_anchoring_candidates(candidates)
        
        # Nettoyage de la description (idem que pour les clusters)
        full_context = " ".join(set(row.description)) if isinstance(row.description, list) else str(row.description)
        clean_context = full_context.replace("\n", " ").strip()
        entity_context = clean_context[:250] + "..." if len(clean_context) > 250 else clean_context

        # 2. Appel au LLM via ask_json
        try:
            result = await self.llm_service.ask_json(
                system_prompt=ANCHORING_RESOLUTION_SYSTEM_PROMPT,
                user_prompt=ANCHORING_RESOLUTION_USER_PROMPT.format(
                    entity_title=row.title,
                    entity_type=row.type,
                    entity_context=entity_context,
                    candidates_text=candidates_text
                )
            )
            return result
        except Exception as e:
            print(f"❌ Erreur LLMResolver sur l'anchoring de {row.title}: {e}")
            # Fallback de sécurité : on considère que c'est une nouvelle entité pour ne pas tout bloquer
            return {"choice": "NEW_ENTITY"}



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

    
    def _format_anchoring_candidates(self, candidates: list) -> str:
        return "\n".join([f"- ID: {c.get('ID', 'UNKNOWN')} | Name: {c.get('CANONICAL_NAME', 'UNKNOWN')} | Summary: {c.get('CORE_SUMMARY', 'No summary.')}" for c in candidates])