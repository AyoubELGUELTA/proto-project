

import asyncio
import logging
import pandas as pd
from typing import List, Union, Tuple
from app.services.llm.service import LLMService
from app.core.config.graph_config import MAX_SUMMARY_LENGTH, ENTITY_BATCH_SIZE, MAX_INPUT_TOKENS #TODO

logger = logging.getLogger(__name__)

class SummarizeManager: 
    def __init__(self, llm_service: LLMService, num_threads: int = ENTITY_BATCH_SIZE):
        self.llm = llm_service
        self.semaphore = asyncio.Semaphore(num_threads)

    async def summarize_all(self, entities_df: pd.DataFrame, relationships_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """L'unique point d'entrée pour nettoyer toutes les descriptions."""
        
        # On traite les deux en parallèle pour gagner du temps
        ent_task = self._process_df(entities_df, is_entity=True)
        rel_task = self._process_df(relationships_df, is_entity=False)
        
        return await asyncio.gather(ent_task, rel_task)

    async def _process_df(self, df: pd.DataFrame, is_entity: bool) -> pd.DataFrame:
        if df.empty:
            return df

        tasks = []
        for row in df.itertuples():
            # Préparation de l'identifiant pour le log/prompt
            if is_entity:
                identifier = row.title
            else:
                identifier = f"{row.source} -> {row.target}"
            
            tasks.append(self._throttled_summarize(identifier, row.description))
        
        df["description"] = await asyncio.gather(*tasks)
        return df

    async def _throttled_summarize(self, identifier: str, descriptions: List[str]) -> str:
        """La logique coeur avec sémaphore."""
        if not descriptions: return ""
        if len(descriptions) == 1 and isinstance(descriptions[0], str): return descriptions[0]
        
        async with self.semaphore:
            unique_descriptions = sorted(set(filter(None, descriptions)))
            
            from app.core.prompts.graph_prompts import SUMMARIZE_SYSTEM_PROMPT, SUMMARIZE_USER_PROMPT
            system_p = SUMMARIZE_SYSTEM_PROMPT.format(max_length=MAX_SUMMARY_LENGTH)
            user_p = SUMMARIZE_USER_PROMPT.format(
                entity_name=identifier,
                description_list="\n- ".join(unique_descriptions)
            )
            
            return await self.llm.ask_text(system_prompt=system_p, user_prompt=user_p)