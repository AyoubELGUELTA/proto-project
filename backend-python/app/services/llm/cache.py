
import redis
import json
import hashlib
import logging
from typing import List, Optional, Any

logger = logging.getLogger(__name__)

class LLMCache:
    """
    Système de persistance pour les réponses LLM basé sur Redis.
    """
    
    def __init__(self, redis_url: str):

        try:
            self.client = redis.from_url(redis_url, decode_responses=True)
            # Test de connexion rapide
            self.client.ping()
        except redis.RedisError as e:
            logger.error(f"❌ Impossible de se connecter à Redis : {e}")
            self.client = None

        self.ttl = 3600 * 24 * 7  # 7 jours
            
    def _generate_key(self, messages: List[Any]) -> str:
        """
        Crée une clé unique (empreinte digitale) à partir des messages envoyés au LLM.
        
        La méthode trie les clés pour garantir que le même dictionnaire 
        produit toujours la même clé de cache (déterminisme).
        
        Args:
            messages: Liste des objets messages (System, Human, etc.)
        Returns:
            str: Une chaîne de type 'llm_cache:a1b2c3d4...'
        """
        # On convertit les objets messages en chaînes pour pouvoir les sérialiser
        serialized = json.dumps([str(m) for m in messages], sort_keys=True)
        # Hachage SHA-256 pour obtenir une clé courte et unique
        hash_gen = hashlib.sha256(serialized.encode()).hexdigest()
        return f"llm_cache:{hash_gen}"

    def get(self, messages: List[Any]) -> Optional[str]:
        """
        Récupère une réponse en cache si elle existe.
        
        Args:
            messages: La liste des messages qui servent de base à la recherche.
        Returns:
            La réponse textuelle si trouvée (HIT), sinon None (MISS).
        """
        key = self._generate_key(messages)
        try:
            return self.client.get(key)
        except redis.RedisError as e:
            logger.warning(f"⚠️ Erreur de lecture Redis : {e}")
            return None

    def set(self, messages: List[Any], response: str):
        """
        Enregistre une réponse LLM dans Redis avec une expiration automatique.
        
        Args:
            messages: Les messages d'origine (utilisés pour générer la clé).
            response: Le texte brut renvoyé par le LLM à stocker.
        """
        key = self._generate_key(messages)
        try:
            # setex = SET avec EXpiration
            self.client.setex(key, self.ttl, response)
        except redis.RedisError as e:
            logger.warning(f"⚠️ Impossible d'écrire dans le cache Redis : {e}")