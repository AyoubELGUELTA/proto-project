# app/services/llm/service.py
from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

from app.services.llm.client import LLMClient
from app.services.llm.tracker import LLMTracker
from app.services.llm.parser import LLMParser
from app.services.llm.cache import LLMCache

class LLMService:
    """
    Point d'entrée unique (Façade) orchestrant les services LLM.
    
    Centralise le tracking de consommation, la gestion du cache Redis,
    la résilience des appels et le parsing des formats (JSON ou Tuples).
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initialise l'ensemble de la stack LLM.
        
        Args:
            model_name (str): Le nom du modèle par défaut à utiliser.
        """
        self.tracker = LLMTracker()
        self.cache = LLMCache()
        self.parser = LLMParser()
        # Le client reçoit le tracker et le cache pour être autonome
        self.client = LLMClient(
            model_name=model_name, 
            tracker=self.tracker, 
            cache=self.cache
        )

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