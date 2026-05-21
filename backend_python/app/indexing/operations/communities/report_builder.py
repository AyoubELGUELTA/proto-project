import logging
from app.services.llm.service import LLMService
from app.indexing.operations.communities.schemas import CommunityReportSchema
from app.core.prompts.graph_prompts.community_report import COMMUNITY_REPORT_SYSTEM_PROMPT, COMMUNITY_REPORT_USER_PROMPT
logger = logging.getLogger(__name__)

class CommunityReportBuilder:
    def __init__(self, llm_reporter: LLMService):
        self.reporter = llm_reporter

    async def generate_report(self, community_id: str, optimized_context: str) -> CommunityReportSchema:
        """Generates an executive structured community report using Microsoft style formatting."""

        logger.info(f"🤖 Dispatching optimized context to LLM for community: {community_id}")

        return await self.reporter.ask_structured(
            system_prompt=COMMUNITY_REPORT_SYSTEM_PROMPT,
            user_prompt=COMMUNITY_REPORT_USER_PROMPT.format(optimized_context=optimized_context),
            response_model=CommunityReportSchema
        )