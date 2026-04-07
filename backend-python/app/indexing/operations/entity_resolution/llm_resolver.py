import logging
import pandas as pd
from typing import Dict
from app.core.prompts.graph_prompts import ENTITY_RESOLUTION_PROMPT
from app.services.llm.parser import LLMParser 

logger = logging.getLogger(__name__)

class LLMResolver:
    def __init__(self, llm_service):
        self.llm_service = llm_service

    async def resolve_cluster(self, cluster_df: pd.DataFrame, entity_type: str) -> Dict[str, str]:
        """
        Prend un cluster d'entités suspectes et demande au LLM d'identifier les doublons.
        Retourne un mapping : {"NOM_A_REMPLACER": "NOM_CIBLE"}
        """
        if len(cluster_df) < 2:
            return {}

        # 1. Formatage des candidats (Nom + Context court)
        candidates = ""
        for row in cluster_df.itertuples():
            # On prend la première description disponible
            desc = row.description[0][:200] if row.description else "No description"
            candidates += f"- {row.title}: {desc}\n"

        prompt = ENTITY_RESOLUTION_PROMPT.format(
            entity_type=entity_type,
            candidates=candidates
        )

        try:
            raw_response = await self.llm_service.ask(prompt)

            if "<|NO_MERGE|>" in raw_response: # CF graph prompts in core/
                return {}
            
            tuples = LLMParser.to_tuples(raw_response, delimiter="<|>")
            
            mapping = {}
            for t in tuples:
                # Format attendu du prompt MC : ("MERGE" <|> "Original" <|> "Target")
                if len(t) == 3 and t[0].upper() == "MERGE":
                    mapping[t[1]] = t[2]
            
            return mapping
        except Exception as e:
            logger.error(f"Erreur LLM lors de la résolution du cluster {entity_type}: {e}")
            return {}