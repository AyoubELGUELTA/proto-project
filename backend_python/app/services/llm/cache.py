import redis
import json
import hashlib
import logging
from typing import List, Optional, Any

logger = logging.getLogger(__name__)

class LLMCache:
    """
    Persistence layer for LLM responses using Redis.
    
    This caching mechanism ensures that identical requests (System + User prompts) 
    do not trigger redundant LLM calls. It optimizes both execution speed and 
    API costs by storing previous completions for a defined TTL (Time To Live).
    """
    
    def __init__(self, redis_url: str):
        """
        Initializes the Redis client and validates the connection.
        
        Args:
            redis_url (str): The connection string (e.g., 'redis://localhost:6379/0').
        """
        try:
            self.client = redis.from_url(redis_url, decode_responses=True)
            # Connectivity check to ensure the service is reachable
            self.client.ping()
            logger.info(f"💾 Redis Cache connected at {redis_url}")
        except redis.RedisError as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            self.client = None

        # Default TTL: 7 days to balance freshness and cost savings
        self.ttl = 3600 * 24 * 7  
            
    def _generate_key(self, messages: List[Any]) -> str:
        """
        Generates a unique fingerprint (SHA-256) based on the request content.
        
        Args:
            messages (List[Any]): List of message objects or strings sent to the LLM.
            
        Returns:
            str: A prefixed hexadecimal string identifier.
        """
        # Stringify each message object to ensure stability in serialization
        serialized = json.dumps([str(m) for m in messages], sort_keys=True)
        # SHA-256 provides a robust collision-resistant identifier
        hash_gen = hashlib.sha256(serialized.encode()).hexdigest()
        return f"llm_cache:{hash_gen}"

    def get(self, messages: List[Any]) -> Optional[str]:
        """
        Retrieves a cached response if available (Cache HIT).
        
        Args:
            messages (List[Any]): The prompt context used as the lookup key.
            
        Returns:
            Optional[str]: The stored completion text or None on Cache MISS.
        """
        if not self.client:
            return None
            
        key = self._generate_key(messages)
        try:
            cached_res = self.client.get(key)
            if cached_res:
                logger.debug(f"💾 Cache HIT for key: {key[:15]}...")
            return cached_res
        except redis.RedisError as e:
            logger.warning(f"⚠️ Redis Read Error: {e}")
            return None

    def set(self, messages: List[Any], response: str):
        """
        Stores an LLM response in Redis with an automatic expiration.
        
        Args:
            messages (List[Any]): The original prompt context.
            response (str): The raw text completion to be stored.
        """
        if not self.client:
            return
            
        key = self._generate_key(messages)
        try:
            # SETEX atomicity: Sets the value and expiration in a single operation
            self.client.setex(key, self.ttl, response)
            logger.debug(f"✅ Response cached: {key[:15]}...")
        except redis.RedisError as e:
            logger.warning(f"⚠️ Failed to write to Redis cache: {e}")