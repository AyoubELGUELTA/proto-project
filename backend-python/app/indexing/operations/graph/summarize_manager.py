# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

import asyncio
import logging
import pandas as pd
from typing import List, Union, Tuple
from app.services.llm.service import LLMService
from app.core.config.graph_config import MAX_SUMMARY_LENGTH, ENTITY_BATCH_SIZE, MAX_INPUT_TOKENS #TODO
from app.core.prompts.graph_prompts import (
                ENTITY_SUMMARIZE_SYSTEM_PROMPT, 
                RELATIONSHIP_SUMMARIZE_SYSTEM_PROMPT,
                COMMON_SUMMARIZE_USER_PROMPT
            )


logger = logging.getLogger(__name__)

class SummarizeManager: 
    """
    Orchestrates the consolidation of multiple descriptions for entities and relationships.
    
    After the extraction and resolution phases, a single entity might have accumulated 
    several fragmented descriptions from different text units. This manager flattens 
    those fragments into a single, cohesive, and grounded summary using a LLM.
    """
    def __init__(self, llm_service: LLMService, num_threads: int = ENTITY_BATCH_SIZE):
        """
        Initializes the manager with a concurrency semaphore.
        
        Args:
            llm_service: The service used to communicate with the LLM.
            num_threads: Maximum number of concurrent LLM requests allowed (rate limiting).
        """
        self.llm = llm_service
        self.semaphore = asyncio.Semaphore(num_threads)

    async def summarize_all(self, entities_df: pd.DataFrame, relationships_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        High-level entry point to clean and summarize all descriptions in the graph.
        
        Processes entities and relationships concurrently to optimize overall execution time.
        
        Returns:
            A tuple of DataFrames (entities, relationships) with updated 'description' columns.
        """
        
        # On traite les deux en parallèle pour gagner du temps
        ent_task = self._process_df(entities_df, is_entity=True)
        rel_task = self._process_df(relationships_df, is_entity=False)
        
        return await asyncio.gather(ent_task, rel_task)

    async def _process_df(self, df: pd.DataFrame, is_entity: bool) -> pd.DataFrame:
        """
        Iterates over a DataFrame to schedule summarization tasks for each row.
        
        Args:
            df: The DataFrame containing items to summarize.
            is_entity: Boolean flag to determine the identifier naming logic.
        """
        if df.empty:
            return df

        tasks = []
        for row in df.itertuples():
            identifier = row.title if is_entity else f"{row.source} -> {row.target}"
            # Passing 'is_entity' flag to pick the proper prompt
            tasks.append(self._throttled_summarize(identifier, row.description, is_entity))
        
        df["description"] = await asyncio.gather(*tasks)
        return df

    async def _throttled_summarize(self, identifier: str, descriptions: List[str], is_entity: bool) -> str:        
        """
        Core logic for synthesizing descriptions with concurrency control.
        
        Optimization:
        1. Returns early if the description list is empty or already contains a single item.
        2. Deduplicates and sorts the input descriptions to maximize cache hits and 
           ensure output stability.
        3. Wraps the LLM call in an 'async with self.semaphore' block to prevent 
           RateLimitErrors.
        """
       
        if not descriptions: 
            return ""
        
        # Grounding optimization: If only one description exists, no synthesis needed
        if len(descriptions) == 1 and isinstance(descriptions[0], str): 
            return descriptions[0]
        
        async with self.semaphore:
            # deduplicate and sort to ensure deterministic input
            unique_descriptions = sorted(set(filter(None, descriptions)))
            
            # Selection of the prompt according to the nature of the object
            system_p = (ENTITY_SUMMARIZE_SYSTEM_PROMPT if is_entity 
                        else RELATIONSHIP_SUMMARIZE_SYSTEM_PROMPT).format(max_length=MAX_SUMMARY_LENGTH)
            
            user_p = COMMON_SUMMARIZE_USER_PROMPT.format(
                target_name=identifier,
                description_list="\n- ".join(unique_descriptions)
            )
            
            return await self.llm.ask_text(system_prompt=system_p, user_prompt=user_p)