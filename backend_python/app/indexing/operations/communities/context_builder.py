import logging
import re
import json
from typing import Dict, Any, List, Tuple

from app.services.graph.community_service import CommunityService
logger = logging.getLogger(__name__)

class HierarchicalContextBuilder:
    def __init__(self, community_service: CommunityService, max_context_tokens: int = 12000): #TODO Centralize
        self.service = community_service
        self.max_context_tokens = max_context_tokens 

    def _estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3)

    async def build_optimized_context(self, community_id: str) -> Tuple[str, dict]:
        """
        Builds an optimized context with explicit local integer IDs.
        Handles Sub-Communities, Entities, and Relationships sequentially.
        
        Returns a tuple: (context_string, mapping_dictionary_for_downstream_resolving)
        """
        logger.info(f"🔍 [START] Building context for community: {community_id}")
        
        try:
            payload = await self.service.get_raw_community_payload(community_id)
            sub_reports = payload.get("sub_reports", [])
            entities = payload.get("entities", [])
            relationships = payload.get("relationships", [])
            
            context_elements: List[str] = []
            current_tokens = 0
            
            # --- NOTRE DICTIONNAIRE DE MAPPING DE PRODUCTION ENRICHI ---
            id_mapping = {
                "sub_communities": {}, # ex: {1: "uuid-subcomm-abc"} 
                "entities": {},        # ex: {1: "uuid-123"}
                "relationships": {}    # ex: {1: "source_ids_xyz"}
            }
            
            added_entity_global_ids = set()
            # Utile pour retrouver le nom/titre d'une entité à partir de son ID global dans la phase 3
            global_id_to_title = {e["id"]: e["title"] for e in entities}

            # Regex compilée pour nettoyer proprement les balises [Data: ...] des enfants 🧹
            data_tag_regex = re.compile(r'\s*\[Data:\s*.*?\]', re.IGNORECASE)

            # --- PHASE 1 : SUB-REPORTS (AVEC ID NUMÉRIQUE LOCAL) ---
            sub_reports_added = 0
            if sub_reports:
                logger.info(f"🌿 Processing {len(sub_reports)} potential sub-reports for context inclusion...")
                
            for idx, report in enumerate(sub_reports, start=1):
                # 1. Structure du bloc de base alignée sur notre nouveau standard
                report_str = (
                    f"[SUB-COMMUNITY] id: {idx} | title: {report.get('title', 'Untitled')} | "
                    f"summary: {report.get('summary', 'No summary provided')}\n"
                )
                
                # 2. Gestion hybride et ultra-robuste de "findings" vs "findings_json"
                raw_findings = report.get("findings") or report.get("findings_json")
                
                # Sécurité au cas où la base renvoie une chaîne JSON brute non parsée
                if isinstance(raw_findings, str):
                    try:
                        raw_findings = json.loads(raw_findings)
                    except Exception as json_err:
                        logger.warning(f"⚠️ Could not parse findings_json string for sub-report: {json_err}")
                        raw_findings = []

                if isinstance(raw_findings, list) and raw_findings:
                    report_str += "Key Findings:\n"
                    for finding in raw_findings:
                        summary = finding.get('summary', 'No summary')
                        explanation = finding.get('explanation', '')
                        
                        # Évite que le LLM Parent ne lise et ne recopie des hashes globaux des entites et source_ids
                        clean_explanation = data_tag_regex.sub('', explanation).strip()
                        
                        report_str += f"- {summary}: {clean_explanation}\n"
                
                # 3. Validation de la fenêtre de contexte & Mapping
                tokens = self._estimate_tokens(report_str)
                if current_tokens + tokens <= self.max_context_tokens:
                    context_elements.append(report_str)
                    current_tokens += tokens
                    
                    # On stocke le vrai ID/UUID de la sous-communauté en base
                    actual_sub_id = report.get("id") or report.get("community_id", f"unknown_sub_{idx}")
                    id_mapping["sub_communities"][idx] = actual_sub_id
                    sub_reports_added += 1
                else:
                    logger.warning(f"⚠️ Context window saturated at Phase 1. Dropping remaining sub-reports.")
                    break

            # --- PHASE 2 : PACKING DES ENTITÉS ---
            sorted_entities = sorted(entities, key=lambda x: x.get("degree", 0), reverse=True)
            
            entities_added = 0
            for ent_idx, ent in enumerate(sorted_entities, start=1):
                ent_str = f"[ENTITY] id: {ent_idx} | name: {ent['title']} | type: {ent['type']} | desc: {ent['description']}\n"
                tokens = self._estimate_tokens(ent_str)
                
                if current_tokens + tokens <= self.max_context_tokens:
                    context_elements.append(ent_str)
                    current_tokens += tokens
                    
                    added_entity_global_ids.add(ent["id"])
                    id_mapping["entities"][ent_idx] = ent["id"]
                    entities_added += 1
                else:
                    logger.warning(f"⚠️ Context window saturated at Phase 2. Dropping remaining entities.")
                    break

            # --- PHASE 3 : PACKING DES RELATIONS ---
            relations_added = 0
            valid_relationships = [
                r for r in relationships 
                if r.get("source") in added_entity_global_ids or r.get("target") in added_entity_global_ids
            ]

            for rel_idx, rel in enumerate(valid_relationships, start=1):
                source_title = global_id_to_title.get(rel.get("source"), rel.get("source"))
                target_title = global_id_to_title.get(rel.get("target"), rel.get("target"))
                
                rel_str = f"[RELATION] id: {rel_idx} | {source_title} -> {target_title} | desc: {rel.get('description')}\n"
                tokens = self._estimate_tokens(rel_str)
                
                if current_tokens + tokens <= self.max_context_tokens:
                    context_elements.append(rel_str)
                    current_tokens += tokens
                    
                    id_mapping["relationships"][rel_idx] = rel.get("source_ids", "")
                    relations_added += 1
                else:
                    logger.warning(f"⚠️ Context window saturated at Phase 3. Dropping remaining relationships.")
                    break

            logger.info(
                f"📊 Context built successfully: {sub_reports_added} Sub-Reports, "
                f"{entities_added} Entities, {relations_added} Relations. Total Estimated Tokens: {current_tokens}"
            )
            
            return "\n".join(context_elements), id_mapping

        except Exception as e:
            logger.critical(f"💥 CRITICAL ERROR inside build_optimized_context: {e}", exc_info=True)
            raise e