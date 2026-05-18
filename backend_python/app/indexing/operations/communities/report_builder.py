import logging
from typing import Dict, Any, List

from app.services.llm.service import LLMService
from app.core.config.graph_config import SUMMARIZATION_LLM_CONFIG
from app.core.data_model.community_report import CommunityReportSchema
from app.core.prompts.graph_prompts.community_prompts import COMMUNITY_REPORT_SYSTEM_PROMPT, COMMUNITY_REPORT_USER_PROMPT

logger = logging.getLogger(__name__)


class CommunityReportBuilder:
    """
    Operation responsible for turning raw community graph contexts into 
    strongly-typed, validated LLM analysis reports using your domain models.
    """

    def __init__(self, llm_service: LLMService):
        """
        Initializes the report builder with the core LLM service.
        """
        self.llm = llm_service

    async def generate_report(self, community_id: str, raw_context: Dict[str, Any]) -> CommunityReportSchema:
        """
        Processes raw community database records, formats them into a prompt,
        and requests a natively validated structured report from the LLM based on CommunityReport.

        Args:
            community_id (str): The unique identifier of the community (e.g., 'level_0_id_1').
            raw_context (Dict): Contains 'entities' and 'relationships' lists from Neo4j.

        Returns:
            CommunityReportSchema: A validated Pydantic model containing title, summary, and findings.
        """
        logger.info(f"📝 Formatting context for community [{community_id}]...")

        # 1. Text formatting of entities for the prompt
        entities_lines = []
        for e in raw_context.get("entities", []):
            entities_lines.append(
                f"- ID: {e['id']} | Title: {e['title']} | Type: {e['type']} | Description: {e['description']}"
            )
        entities_context = "\n".join(entities_lines) if entities_lines else "No entities in this community."

        # 2. Text formatting of relationships for the prompt
        rels_lines = []
        for r in raw_context.get("relationships", []):
            rels_lines.append(
                f"- {r['source']} ──> {r['target']} : {r['description']}"
            )
        relationships_context = "\n".join(rels_lines) if rels_lines else "No internal relationships."

        # 3. Final assembly of the user prompt
        user_prompt = COMMUNITY_REPORT_USER_PROMPT.format(
            entities_context=entities_context,
            relationships_context=relationships_context
        )

        try:
            logger.info(f"🧠 Dispatching community [{community_id}] to LLM for structured summarization...")
            
            # Utilizing the structured extraction method with your imported schema
            report: CommunityReportSchema = await self.llm.ask_structured(
                system_prompt=COMMUNITY_REPORT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=CommunityReportSchema,
                config=SUMMARIZATION_LLM_CONFIG
            )

            logger.info(f"✅ Validated structured report successfully generated for [{community_id}].")
            return report

        except Exception as e:
            logger.error(f"❌ Failed to generate structured LLM report for community {community_id}: {e}")
            raise