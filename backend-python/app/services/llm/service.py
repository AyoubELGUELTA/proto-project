from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

from app.services.llm.parser import LLMParser
from .factory import LLMFactory
from .parser import LLMParser
from app.core.config.llm_config import LLMConfig

class LLMService:
    def __init__(self, config: LLMConfig = None):
        self.client = LLMFactory.create_client(config)
        self.parser = LLMParser()
        # On accède au tracker via le client pour les rapports
        self.tracker = self.client.tracker

    async def extract_tuples(self, system_prompt: str, user_prompt: str) -> List[List[str]]:
        """
        Exécute une extraction structurée au format GraphRAG (tuples <|>).
        
        Args:
            system_prompt (str): Instructions de rôle et règles d'extraction.
            user_prompt (str): Le contenu (chunk/text unit) à analyser.
            
        Returns:
            List[List[str]]: Une matrice de données extraites (entités ou relations).
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        raw_text = await self.client.ask(messages)
        return self.parser.to_tuples(raw_text)

    async def ask_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Interroge le LLM pour obtenir une réponse strictement formatée en JSON.
        
        Args:
            system_prompt (str): Instructions incluant le schéma JSON attendu.
            user_prompt (str): La question ou les données d'entrée.
            
        Returns:
            Dict[str, Any]: Un dictionnaire Python prêt à l'emploi.
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        raw_text = await self.client.ask(messages)
        return self.parser.to_json(raw_text)

    def get_usage_report(self) -> str:
        """
        Génère un résumé textuel de la consommation de la session actuelle.
        
        Returns:
            str: Rapport incluant tokens et coût estimé.
        """
        return self.tracker.get_report()