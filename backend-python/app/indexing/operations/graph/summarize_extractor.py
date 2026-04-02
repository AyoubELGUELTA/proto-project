from typing import List, Union, Tuple
import json
from app.services.llm.client import LLMClient
from app.core.config.graph_config import SummarizationConfig

class SummarizeExtractor: 
    def __init__(self, llm_client: LLMClient, config: SummarizationConfig, prompt_template: str):
        self.llm = llm_client
        self.config = config
        self.prompt_template = prompt_template

    async def __call__(self, id: Union[str, Tuple[str, str]], descriptions: List[str]) -> str:
        if not descriptions: 
            return ""
        if len(descriptions) == 1: 
            return descriptions[0]
        
        unique_descriptions = sorted(set(descriptions))
        
        prompt = self.prompt_template.format(
            entity_name=json.dumps(id, ensure_ascii=False),
            description_list=json.dumps(unique_descriptions, ensure_ascii=False),
            max_length=self.config.max_summary_length
        )
        
        # Formatage des messages pour le LLMClient
        messages = [{"role": "user", "content": prompt}]
        return await self.llm.ask(messages)