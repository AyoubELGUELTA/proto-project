import logging
from typing import List, Union, Tuple
from app.services.llm.service import LLMService
from app.core.config.graph_config import MAX_SUMMARY_LENGTH
from app.core.prompts.graph_prompts import SUMMARIZE_SYSTEM_PROMPT, SUMMARIZE_USER_PROMPT

logger = logging.getLogger(__name__)

class SummarizeExtractor: 
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    async def __call__(self, identifier: Union[str, Tuple[str, str]], descriptions: List[str]) -> str:
        if not descriptions: 
            return ""
        if len(descriptions) == 1: 
            return descriptions[0]
        
        # Déduplication et tri pour la stabilité du cache
        unique_descriptions = sorted(set(descriptions))
        
        # Préparation des prompts via nos constantes importées
        system_p = SUMMARIZE_SYSTEM_PROMPT.format(max_length=MAX_SUMMARY_LENGTH)
        user_p = SUMMARIZE_USER_PROMPT.format(
            entity_name=str(identifier),
            description_list="\n- ".join(unique_descriptions)
        )
        
        return await self.llm.ask_text(system_prompt=system_p, user_prompt=user_p)
    
