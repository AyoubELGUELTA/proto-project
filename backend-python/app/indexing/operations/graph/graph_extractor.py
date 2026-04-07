# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

import logging

from app.core.config.graph_config import ExtractionConfig 
from app.services.llm.client import LLMClient
from app.core.prompts.graph_prompts import GRAPH_EXTRACTION_PROMPT, CONTINUE_PROMPT, LOOP_PROMPT

logger = logging.getLogger(__name__)


class GraphExtractor:
    def __init__(self, llm_client: LLMClient, config: ExtractionConfig):
        self.llm = llm_client
        self.config = config

    async def __call__(self, text: str, context: str) -> str:
        """Point d'entrée unique : Extrait le vrac brut d'un chunk."""
        return await self._extract_with_gleaning(text, context)

    async def _extract_with_gleaning(self, text: str, context: str) -> str:
        """Boucle de Gleaning calquée sur Microsoft (Inchangée sur le fond)."""
        prompt_input = GRAPH_EXTRACTION_PROMPT.format(
            entity_types=",".join(self.config.entity_types),
            document_metadata=context,
            input_text=text
        )

        messages = [{"role": "user", "content": prompt_input}]
        response = await self.llm.ask(messages)
        full_results = response
        messages.append({"role": "assistant", "content": response})

        if self.config.max_gleanings > 0:
            for i in range(self.config.max_gleanings):
                messages.append({"role": "user", "content": CONTINUE_PROMPT})
                response = await self.llm.ask(messages)
                full_results += f"\n{self.config.record_delimiter}\n{response}"
                
                if i >= self.config.max_gleanings - 1: break

                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": LOOP_PROMPT})
                check = await self.llm.ask(messages)
                if "Y" not in check.upper(): break
        
        return full_results