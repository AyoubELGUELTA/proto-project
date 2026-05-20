import logging
import hashlib
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class CommunityStateEvaluator:
    """
    Utility class responsible for evaluating structural and content updates
    within a graph community to optimize LLM reporting budgets.
    """

    @staticmethod
    def generate_fingerprint(entity_count: int, relationship_count: int, semantic_mass: int) -> str:
        """
        Creates a deterministic hash representing the exact state of the community's data.
        If counts change, or if a single node/edge is updated (modifying timestamps), the hash changes.
        """
        state_string = f"e_cnt:{entity_count}|r_cnt:{relationship_count}|sm:{semantic_mass}"
        return hashlib.md5(state_string.encode("utf-8")).hexdigest()

    @staticmethod
    def evaluate_divergence(state: Dict[str, Any], threshold: float = 0.05) -> Tuple[str, float]: #TODO centralize the threshold value
        """
        Compares current community metrics against the last recorded report state.
        
        Args:
            state (Dict): Combined metrics from Neo4j.
            threshold (float): Max percentage of allowed structural drift before forcing a rewrite (default 15%).
            
        Returns:
            Tuple[str, float]: (Strategy to execute ('SKIP', 'CREATE', 'REWRITE'), divergence_score (between 0 and 1))
        """
        if state["last_report_hash"] is None:
            return "CREATE", 1.0

        if state["current_hash"] == state["last_report_hash"]:
            return "SKIP", 0.0

        # 1. Extraction des anciennes baselines
        prev_entities = state["last_report_entity_count"] or 0
        prev_rels = state["last_report_relationship_count"] or 0
        prev_mass = state["last_report_semantic_mass"] or 0

        # 2. Calcul des ratios de dérive individuels (bornés à 1.0 max pour éviter les explosions)
        drift_e = min(abs(state["current_entity_count"] - prev_entities) / max(prev_entities, 1), 1.0)
        drift_r = min(abs(state["current_relationship_count"] - prev_rels) / max(prev_rels, 1), 1.0)
        drift_m = min(abs(state["current_semantic_mass"] - prev_mass) / max(prev_mass, 1), 1.0)

        # 3. Application des poids relatifs (Somme des poids = 1.0)
        w_entities = 0.50  # Entities are the most considerable factor
        w_relations = 0.30 # The network topology
        w_mass = 0.20      # The textual information density 
        divergence_score = (w_entities * drift_e) + (w_relations * drift_r) + (w_mass * drift_m)

        # 4. Prise de décision
        if divergence_score >= threshold:
            return "REWRITE", divergence_score
        
        return "SKIP", divergence_score