import logging
from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

from app.services.llm.client import LLMClient
from app.services.llm.parser import LLMParser

logger = logging.getLogger(__name__)

class LLMService:
    """
    High-level service orchestrating LLM interactions and data parsing.
    
    It manages the transition from raw text prompts to structured Python objects 
    (JSON, Tuples) and tracks token usage across the session.
    """
    
    def __init__(self, client: LLMClient):
        """
        Initializes the service with a specialized client and parser.

        Args:
            client (LLMClient): The underlying client handling API calls (OpenAI/Anthropic).
        """
        self.client = client
        self.parser = LLMParser()
        self.tracker = client.tracker

    async def ask_tuples(self, system_prompt: str, user_prompt: str) -> List[List[str]]:
        """
        Executes a structured extraction using the MC GraphRAG tuple format (<|>).

        Args:
            system_prompt (str): Instructions defining the extraction schema and role.
            user_prompt (str): The raw text content to be analyzed.

        Returns:
            List[List[str]]: A list of extracted records, where each record is a list of strings 
                             (e.g., [["entity", "name", "type", "description"], ["relation", "source", "target",...]]).
                             Returns an empty list if parsing fails.
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        raw_text = await self.client.ask(messages)
        
        try:
            tuples = self.parser.to_tuples(raw_text)
            if not tuples:
                logger.warning("⚠️ LLM returned empty or malformed tuples.")
            return tuples
        except Exception as e:
            logger.error(f"❌ Failed to parse LLM response into tuples: {e}")
            logger.debug(f"Raw problematic output: {raw_text[:200]}...")
            return []

    async def ask_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Queries the LLM for a strictly formatted JSON response.

        Args:
            system_prompt (str): Instructions specifying the required JSON structure.
            user_prompt (str): The context or question to be processed.

        Returns:
            Dict[str, Any]: The parsed JSON data as a Python dictionary.
                            Returns an empty dictionary {} if parsing fails.
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        raw_text = await self.client.ask(messages)
        
        try:
            return self.parser.to_json(raw_text)
        except Exception as e:
            logger.error(f"❌ JSON Parsing error: {e}")
            # We log the first part of the raw text to help debug the format issue
            logger.debug(f"Faulty JSON raw text: {raw_text[:500]}")
            return {}
    
    async def ask_text(self, system_prompt: str, user_prompt: str) -> str:
        """
        Queries the LLM for a simple natural language response.

        Args:
            system_prompt (str): Context or persona instructions.
            user_prompt (str): The query or text to summarize/process.

        Returns:
            str: The raw completion text from the LLM.
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        return await self.client.ask(messages)
        
    def _tuples_to_string(self, tuples: List[List[str]], delimiter: str = "##") -> str:
        """
        Reconstructs a raw string from tuples for LLM context history or debugging.

        Args:
            tuples (List[List[str]]): The data records to format.
            delimiter (str): The separator between records. Defaults to "##".

        Returns:
            str: A formatted string: (val1<|>val2) ## (val3<|>val4).
        """
        formatted_list = []
        for t in tuples:
            inner = "<|>".join(str(item) for item in t)
            formatted_list.append(f"({inner})")
        
        return f"\n{delimiter}\n".join(formatted_list)

    def get_usage_report(self) -> str:
        """
        Generates a summary of the current session's token consumption.

        Returns:
            str: A human-readable report including prompt tokens, completion tokens, and estimated cost.
        """
        report = self.tracker.get_report()
        logger.info("📊 Session Usage Report generated.")
        return report