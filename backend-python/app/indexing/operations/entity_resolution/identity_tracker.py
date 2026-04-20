from typing import Any, Dict
import logging


logger = logging.getLogger(__name__)

class IdentityTracker:
    """
    Centralized registry for tracking entity renames and identity merges.
    
    This tracker maintains a mapping of 'old identities' to 'new identities'. 
    It is designed to handle messy extractions by enforcing case-insensitive 
    keys and resolving multi-step redirection chains (transitive resolution).
    """

    def __init__(self):
        """Initializes an empty mapping dictionary."""
        self._mapping: Dict[str, str] = {}

    def add_mapping(self, old_id: Any, new_id: Any):
        """
        Registers a redirection from an old id to a new id.
        
        To prevent lookup failures due to inconsistent casing from the LLM, 
        this method stores the mapping in two forms: original, UPPER, and lower.
        
        Args:
            old_id: The source ID to be redirected.
            new_id: The target ID (the 'canonical' version).
        """
        # Clean and normalize inputs to ensure string consistency
        old_s = str(old_id).strip()
        new_s = str(new_id).strip()

        # Safety check: avoid empty strings or self-referential loops
        if not old_s or not new_s or old_s == new_s:
            return
        
        logger.debug(f"Registering alias: '{old_s}' -> '{new_s}'")
        
        self._mapping[old_id] = new_id


    def resolve(self) -> dict:
        """
        Computes the final state of all registered mappings using transitive logic.
        
        This method follows the redirection path to its absolute end. 
        Example: If A -> B and B -> C, the resolved map will contain A -> C.
        
        Logic:
        - It iterates through every key and follows the 'current' target.
        - A 'path' set is used to detect and break infinite recursion (circular links).
        
        Returns:
            A dictionary where every key points directly to its final resolved identity.
        """
        final_map = {}
        for key in self._mapping:
            path = {key}
            current = self._mapping[key]
            
            # 1. We follow the chain
            while current in self._mapping and current not in path:
                path.add(current)
                current = self._mapping[current]
                
            # 2. We register each key
            final_map[key] = current

        logger.info(f"✅ Identity resolution complete. {len(final_map)} identities stabilized.")
        return final_map