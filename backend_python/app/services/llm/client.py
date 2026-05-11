# app/services/llm/client.py
import logging
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from app.services.llm.tracker import LLMTracker
from app.services.llm.cache import LLMCache
from app.core.config.llm_config import LLMConfig
from app.core.settings import settings

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Core execution engine for LLM interactions.
    
    This client acts as the gateway to Model APIs (OpenAI,Gemini,etc), integrating:
    1. Resilience: Automatic retries with exponential backoff.
    2. Efficiency: Redis-based response caching to prevent redundant API calls.
    3. Monitoring: Token usage tracking and cost estimation.
    """

    def __init__(self, config: LLMConfig, api_key: str, tracker: LLMTracker, cache: LLMCache):
        """
        Initializes the client with its required infrastructure.
        
        Args:
            config: Configuration object defining model parameters (model_name, temp, max_retries, etc.).
            api_key: Secret key for API authentication.
            tracker: Instance of LLMTracker for consumption monitoring.
            cache: Instance of LLMCache for persistence.
        """
        self.config = config 
        self.tracker = tracker
        self.cache = cache
        self.model_name = config.model_name 

        # Integration with LangChain's ChatOpenAI abstraction

        self.llm = ChatOpenAI(
            model=self.model_name, 
            openai_api_key=api_key, 
            temperature=config.temperature,
            max_retries=config.max_retries,
            streaming=config.streaming,
            stream_usage=config.token_report
        )

    async def ask(self, messages: list) -> str:
        """
        Main entry point for LLM requests.
        
        Orchestrates the 'Cache-First' strategy:
        - If the request fingerprint exists in Redis, returns immediately.
        - Otherwise, triggers a resilient network call and caches the result.

        Args:
            messages: List of LangChain message objects (SystemMessage, HumanMessage).

        Returns:
            The textual content of the model's response.
        """

        # 1. Cache Lookup (Saves money and time)
        cached_response = self.cache.get(messages)
        if cached_response:
            print(f"💾 Cache HIT for model {self.model_name}")
            return cached_response

        # 2. Resilient API Call
        print(f"🌐 Cache MISS. Dispatching API call to {self.model_name}...")
        response_text = await self._execute_with_retry(messages)

        # 3. Cache Update
        self.cache.set(messages, response_text)
        
        return response_text


    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True
    )
    async def _execute_with_retry(self, messages: list) -> str:
        """
        Executes the network call with a safety retry mechanism.
        
        The 'exponential backoff' ensures that the client waits progressively 
        longer (2s, 4s, 8s...) between attempts to handle rate limits or transient errors.

        Args:
            messages: List of messages formatted for LangChain.

        Returns:
            Raw completion content.
            
        Raises:
            Exception: If the call fails after the maximum number of attempts.
        """
        # Await the asynchronous LangChain call
        response = await self.llm.ainvoke(messages)
        
        # Metadata Extraction (Handling variations between LangChain versions)
        usage = getattr(response, "usage_metadata", {}) or {}
        resp_meta = getattr(response, "response_metadata", {}) or {}
        legacy_usage = resp_meta.get("token_usage", {}) or {}

        # Resolve token counts from multiple possible metadata locations
        prompt_tokens = (
            usage.get("input_tokens") 
            or legacy_usage.get("prompt_tokens") 
            or 0
        )
        completion_tokens = (
            usage.get("output_tokens") 
            or legacy_usage.get("completion_tokens") 
            or 0
        )

        # Usage Tracking
        if self.tracker:
            self.tracker.add_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model_name=self.model_name
            )
        
        return response.content