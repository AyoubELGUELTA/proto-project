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
        comm_id = state.get("id", "UNKNOWN_ID")
        last_hash = state.get("last_report_hash")
        curr_hash = state.get("current_hash")

        # 🚨 LOG DE DIAGNOSTIC INITIAL
        logger.info(
            f"🕵️ Inspecting drift for community {comm_id} -> "
            f"current_hash: '{curr_hash}' | last_report_hash: '{last_hash}'"
        )

        # Cas 1 : Absence totale de rapport historique
        if last_hash is None or last_hash == "" or last_hash == "null":
            logger.info(f"🆕 [DECISION] -> CREATE for community {comm_id} (Reason: No historical report hash found).")
            return "CREATE", 1.0

        # Cas 2 : Identité parfaite (Aucun changement structurel)
        if curr_hash == last_hash:
            logger.info(f"😴 [DECISION] -> SKIP for community {comm_id} (Reason: Current hash perfectly matches historical hash).")
            return "SKIP", 0.0

        # 1. Extraction des anciennes baselines
        prev_entities = state.get("last_report_entity_count") or 0
        prev_rels = state.get("last_report_relationship_count") or 0
        prev_mass = state.get("last_report_semantic_mass") or 0

        # 2. Calcul des ratios de dérive individuels
        drift_e = min(abs(state.get("current_entity_count", 0) - prev_entities) / max(prev_entities, 1), 1.0)
        drift_r = min(abs(state.get("current_relationship_count", 0) - prev_rels) / max(prev_rels, 1), 1.0)
        drift_m = min(abs(state.get("current_semantic_mass", 0) - prev_mass) / max(prev_mass, 1), 1.0)

        # 3. Application des poids relatifs
        w_entities = 0.50  
        w_relations = 0.30 
        w_mass = 0.20      
        divergence_score = (w_entities * drift_e) + (w_relations * drift_r) + (w_mass * drift_m)

        # 4. Prise de décision basée sur le score
        if divergence_score >= threshold:
            logger.info(f"🔄 [DECISION] -> REWRITE for community {comm_id} (Score: {divergence_score:.4f} >= Threshold: {threshold}).")
            return "REWRITE", divergence_score
        
        logger.info(f"💤 [DECISION] -> SKIP for community {comm_id} (Score: {divergence_score:.4f} < Threshold: {threshold}).")
        return "SKIP", divergence_score