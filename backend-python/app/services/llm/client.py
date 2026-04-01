# app/services/llm/client.py
import logging
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from app.services.llm.tracker import LLMTracker
from app.services.llm.cache import LLMCache

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Moteur d'exécution LLM gérant la résilience (Retries), le Caching et le Tracking.
    
    Cette classe est le seul point de contact direct avec les APIs de modèles (OpenAI/Gemini).
    """

    def __init__(self, model_name: str, tracker: LLMTracker, cache: LLMCache, temperature: float = 0):
        """
        Args:
            model_name (str): Identifiant du modèle (ex: 'gpt-4o-mini').
            tracker (LLMTracker): Instance pour le suivi des coûts et tokens.
            cache (LLMCache): Instance Redis pour le cache des réponses.
            temperature (float): Degré de créativité du modèle (0 pour l'extraction).
        """
        self.model_name = model_name
        self.tracker = tracker
        self.cache = cache
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)

    async def ask(self, messages: list) -> str:
        """
        Point d'entrée principal pour envoyer une requête au LLM.
        Vérifie d'abord le cache avant de déclencher un appel réseau résilient.

        Args:
            messages (list): Liste d'objets SystemMessage ou HumanMessage.

        Returns:
            str: Le contenu textuel de la réponse du modèle.
        """
        # 1. Tentative de récupération depuis le cache Redis
        cached_response = self.cache.get(messages)
        if cached_response:
            logger.info(f"💾 Cache HIT pour le modèle {self.model_name}")
            return cached_response

        # 2. Appel réseau avec logique de Retry intégrée
        logger.info(f"🌐 Cache MISS. Appel API vers {self.model_name}...")
        response_text = await self._execute_with_retry(messages)

        # 3. Mise en cache de la nouvelle réponse
        self.cache.set(messages, response_text)
        
        return response_text

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True
    )
    async def _execute_with_retry(self, messages: list) -> str:
        """
        Exécute l'appel réseau vers le LLM avec un mécanisme de backoff exponentiel.

        Args:
            messages (list): Liste de messages formatés pour LangChain.

        Returns:
            str: Contenu brut de la réponse.
            
        Raises:
            Exception: Si l'appel échoue après 3 tentatives.
        """
        response = await self.llm.ainvoke(messages)
        
        # Extraction et enregistrement de la consommation
        meta = response.response_metadata.get("token_usage", {})
        self.tracker.add_usage(
            prompt_tokens=meta.get("prompt_tokens", 0),
            completion_tokens=meta.get("completion_tokens", 0),
            model_name=self.model_name
        )
        
        return response.content