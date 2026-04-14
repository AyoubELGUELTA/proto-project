# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License


from typing import List
from app.core.config.graph_config import ENTITY_TYPES, MAX_GLEANINGS, RECORD_DELIMITER
from app.services.llm.service import LLMService
from app.core.prompts.graph_prompts import (
    GRAPH_EXTRACTION_SYSTEM_PROMPT, 
    GRAPH_EXTRACTION_USER_PROMPT,
    CONTINUE_PROMPT, 
    LOOP_PROMPT
)


class EntityAndRelationExtractor:
    """
    Handles the zero-shot extraction of entities and relationships from text using an iterative 'gleaning' process.
    
    This extractor uses a LLM to transform unstructured text into structured tuples (nodes and edges).
    It implements a feedback loop to ensure maximum recall of information that might be missed 
    in a single pass.
    """

    def __init__(self, llm_service: LLMService):
        """Initializes the extractor with a specialized LLM service for tuple generation."""
        self.llm = llm_service 

    async def __call__(self, text: str, context: str) -> List[List[str]]:
        """
        Entry point to extract raw graph tuples from a text unit.
        
        Args:
            text: The raw text content to analyze.
            context: Domain-specific metadata or summaries to guide the LLM's focus.
            
        Returns:
            A list of raw tuples, where each tuple represents an entity or a relationship. 
            (cf graph_prompts to look at the output models)
        """
        return await self._extract_with_gleaning(text, context)

    async def _extract_with_gleaning(self, text: str, context: str) -> List[List[str]]:

        """
        Executes an iterative extraction process to minimize information loss.
        
        The process follows these steps:
        1. Initial extraction of visible entities/relationships.
        2. 'Gleaning' cycles: Probing the LLM's conversation history to find 
           missed information until a limit is reached or the LLM signals completion (LOOP_PROMPT).
        
        This multi-turn approach is critical for dense texts where a single response 
        might hit token limits or overlook subtle connections.
        """

        # 1. Prompts (Cache-friendly)
        sys_p = GRAPH_EXTRACTION_SYSTEM_PROMPT.format(entity_types=",".join(ENTITY_TYPES),document_metadata=context)
        usr_p = GRAPH_EXTRACTION_USER_PROMPT.format(
            entity_types=",".join(ENTITY_TYPES),
            input_text=text
        )

        # 2. Premier passage
        all_tuples = await self.llm.ask_tuples(system_prompt=sys_p, user_prompt=usr_p)

        # 3. Gleaning
        if MAX_GLEANINGS > 0:
            history = [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": usr_p},
                {"role": "assistant", "content": self.llm._tuples_to_string(RECORD_DELIMITER, all_tuples)}
            ]

            for i in range(MAX_GLEANINGS):
                history.append({"role": "user", "content": CONTINUE_PROMPT})
                raw_res = await self.llm.client.ask(history)
                
                new_tuples = self.llm.parser.to_tuples(raw_res)
                if not new_tuples: break
                
                all_tuples.extend(new_tuples)
                if i >= MAX_GLEANINGS - 1: break

                history.extend([
                    {"role": "assistant", "content": raw_res},
                    {"role": "user", "content": LOOP_PROMPT}
                ])
                if "Y" not in (await self.llm.client.ask(history)).upper(): break
        
        return all_tuples