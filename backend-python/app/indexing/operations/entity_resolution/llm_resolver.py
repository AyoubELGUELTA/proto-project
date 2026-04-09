import pandas as pd
from typing import Dict
from app.services.llm.service import LLMService
from app.core.prompts.graph_prompts import (
    ENTITY_RESOLUTION_SYSTEM_PROMPT, 
    ENTITY_RESOLUTION_USER_PROMPT
)


class LLMResolver:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

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