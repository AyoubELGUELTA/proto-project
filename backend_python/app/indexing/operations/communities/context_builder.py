import logging
from typing import Dict, Any, List

from app.services.graph.community_service import CommunityService
logger = logging.getLogger(__name__)

class HierarchicalContextBuilder:
    def __init__(self, community_service : CommunityService, max_context_tokens: int = 12000): 
        self.service = community_service
        self.max_context_tokens = max_context_tokens

    def _estimate_tokens(self, text: str) -> int:
        # Approximation classique de production : ~1.3 tokens par mot en moyenne
        return int(len(text.split()) * 1.3)

    async def build_optimized_context(self, community_id: str) -> str:
        """
        Builds a high-fidelity hybrid context matching Microsoft's build_mixed_context.
        Prioritizes dense sub-reports, then packs direct 'star' entities and internal relationships
        until the token budget is fully depleted.
        """
        # Isolation de la couche de données grâce à ton refactor
        payload = await self.service.get_raw_community_payload(community_id)
        
        context_elements: List[str] = []
        current_tokens = 0
        added_entity_ids = set()

        # 1. Priorité 1 : Les rapports des sous-communautés enfants (Compression maximale)
        sub_reports = payload.get("sub_reports", [])
        for report in sub_reports:
            report_str = (
                f"[SUB-COMMUNITY REPORT {report['id']}]\n"
                f"Title: {report['title']}\n"
                f"Summary: {report['summary']}\n"
            )
            
            # Optionnel : On injecte les findings des enfants s'ils existent
            if report.get("findings"):
                report_str += "Key Findings:\n"
                for finding in report["findings"]:
                    report_str += f"- {finding.get('summary')}: {finding.get('explanation')}\n"
            
            tokens = self._estimate_tokens(report_str)
            
            if current_tokens + tokens <= self.max_context_tokens:
                context_elements.append(report_str)
                current_tokens += tokens
            else:
                # C'est l'élagage déterministe de Microsoft : dès que ça déborde, on loggue et on passe au tamis suivant
                logger.warning(
                    f"⚠️ Token budget reached during sub-reports packing for {community_id}. "
                    f"Truncating remaining sub-reports."
                )
                break  # On arrête d'ajouter des gros blocs de rapports

        # 2. Priorité 2 : Les entités directes orphelines (Triées par importance)
        # Note sur ton TODO : En Neo4j, "degree" est correct si tu as exécuté un algo de centralité, 
        # mais utiliser .get() avec un fallback à 0 sécurise le code contre les KeyErrors.
        sorted_entities = sorted(
            payload.get("entities", []), 
            key=lambda x: x.get("degree", 0), 
            reverse=True
        )
        
        for ent in sorted_entities:
            ent_str = f"[DIRECT ENTITY] {ent['title']} ({ent['type']}) - {ent['description']}\n"
            tokens = self._estimate_tokens(ent_str)
            
            if current_tokens + tokens <= self.max_context_tokens:
                context_elements.append(ent_str)
                current_tokens += tokens
                added_entity_ids.add(ent["id"])
            else:
                # Le sac à dos est plein pour les entités, on s'arrête là
                break

        # 3. Priorité 3 : Les relations internes associées aux entités conservées
        relationships = payload.get("relationships", [])
        for rel in relationships:
            # On ne prend la relation que si elle implique une entité star qu'on a réussi à caser
            if rel.get("source") in added_entity_ids or rel.get("target") in added_entity_ids:
                rel_str = f"[RELATION] {rel.get('source')} -> {rel.get('target')} | Desc: {rel.get('description')}\n"
                tokens = self._estimate_tokens(rel_str)
                
                if current_tokens + tokens <= self.max_context_tokens:
                    context_elements.append(rel_str)
                    current_tokens += tokens
                else:
                    break # Budget saturé à 100%

        logger.info(f"📊 Final context built for {community_id} with {current_tokens}/{self.max_context_tokens} tokens.")
        return "\n".join(context_elements)