# app/services/llm/client.py
import logging
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
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

    def __init__(self, config: LLMConfig, tracker: LLMTracker, cache: LLMCache, api_keys: Dict[str, str]):
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
        self.provider = config.provider.lower()
        self.api_keys = api_keys

        # Integration with LangChain's ChatOpenAI abstraction

        self.llm = self._build_underlying_llm(
            provider=self.provider,
            model_name=self.model_name,
            temperature=config.temperature,
            max_retries=config.max_retries,
            streaming=config.streaming,
            token_report=config.token_report
        )

    def _build_underlying_llm(self, provider: str, model_name: str, temperature: float, 
                              max_retries: int, streaming: bool, token_report: bool):
        """Construit l'instance de ChatModel LangChain appropriée avec les bonnes clés d'API."""
        
        if provider == "openai":
            return ChatOpenAI(
                model=model_name, 
                openai_api_key=self.api_keys.get("openai"), 
                temperature=temperature,
                max_retries=max_retries,
                streaming=streaming,
                stream_usage=token_report
            )
            
        elif provider == "anthropic":
            return ChatAnthropic(
                model_name=model_name,
                anthropic_api_key=self.api_keys.get("anthropic"),
                temperature=temperature,
                max_retries=max_retries,
                streaming=streaming
            )
            
        elif provider == "deepseek":
            return ChatDeepSeek(
                model=model_name,
                api_key=self.api_keys.get("deepseek"),
                base_url="https://api.deepseek.com/v1",
                temperature=temperature,
                max_retries=max_retries,
                streaming=streaming,
                stream_usage=token_report
            )
            
        elif provider == "mistral":
            return ChatMistralAI(
                model=model_name,
                mistral_api_key=self.api_keys.get("mistral"),
                temperature=temperature,
                max_retries=max_retries,
                streaming=streaming
            )
        
        elif provider == "google":
            return ChatGoogleGenerativeAI(
                model=model_name,
                api_key=self.api_keys.get("google"),
                temperature=temperature,
                max_retries=max_retries,
                streaming=streaming
            )
            
        else:
            raise ValueError(f"❌ Unsupported LLM Provider requested: '{provider}'")

    async def ask(self, messages: list, response_format : dict = None, config : Any = None) -> str:
        """
        Main entry point for LLM requests.
        
        Orchestrates the 'Cache-First' strategy:
        - If the request fingerprint exists in Redis, returns immediately.
        - Otherwise, triggers a resilient network call and caches the result.

        Args:
            messages: List of LangChain message objects (SystemMessage, HumanMessage).
            response_format (dict, optional): Native JSON Schema format for strict outputs.
            config (Any, optional): Model-specific configuration override (e.g. SUMMARIZATION_LLM_CONFIG).

        Returns:
            The textual content of the model's response.
        """
        # 1. Cache Lookup
        cached_response = self.cache.get(messages)
        if cached_response:
            print(f"💾 Cache HIT for model {self.model_name}")
            return cached_response

        # 2. Resilient API Call (We pass down both response_format and config)
        print(f"🌐 Cache MISS. Dispatching API call to {self.model_name}...")
        response_text = await self._execute_with_retry(
            messages, 
            response_format=response_format, 
            config=config
        )

        # 3. Cache Update
        self.cache.set(messages, response_text)
        
        return response_text


    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True
    )
    async def _execute_with_retry(self, messages: list, response_format: dict = None, config: Any = None) -> str:
        """
        Executes the network call with a safety retry mechanism and runtime configuration overrides.
        
        The 'exponential backoff' ensures that the client waits progressively 
        longer (2s, 4s, 8s...) between attempts to handle rate limits or transient errors.

        Args:
            messages: List of messages formatted for LangChain.
            response_format (dict, optional): Native JSON Schema format for strict outputs.
            config (Any, optional): Model-specific configuration override.

        Returns:
            Raw completion content.
            
        Raises:
            Exception: If the call fails after the maximum number of attempts.
        """
        # Build base generation arguments
        kwargs = {}
        
        # Le format de réponse strict (JSON Mode) est supporté nativement par OpenAI, DeepSeek et Mistral
        if response_format and self.provider in ["openai", "deepseek", "mistral", "google"]:
            kwargs["response_format"] = response_format

        current_model_name = self.model_name
        
        # Gestion dynamique des surcharges de configuration au runtime
        if config:
            if isinstance(config, dict):
                if "temperature" in config:
                    kwargs["temperature"] = config["temperature"]
                if "model_name" in config:
                    kwargs["model"] = config["model_name"]
                    current_model_name = config["model_name"]
            else:
                if hasattr(config, "temperature"):
                    kwargs["temperature"] = config.temperature
                if hasattr(config, "model_name"):
                    # Gestion de l'asymétrie de nommage de variable selon les providers dans ainvoke
                    param_key = "model"
                    kwargs[param_key] = config.model_name
                    current_model_name = config.model_name

        # Invocation unifiée de LangChain
        response = await self.llm.ainvoke(messages, **kwargs)
        
        # Extraction et normalisation standardisée des jetons multi-provider
        usage = getattr(response, "usage_metadata", {}) or {}
        resp_meta = getattr(response, "response_metadata", {}) or {}
        legacy_usage = resp_meta.get("token_usage", {}) or {}

        prompt_tokens = usage.get("input_tokens") or legacy_usage.get("prompt_tokens") or 0
        completion_tokens = usage.get("output_tokens") or legacy_usage.get("completion_tokens") or 0

        if self.tracker:
            self.tracker.add_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model_name=f"{self.provider}/{current_model_name}"
            )
        
        return response.content