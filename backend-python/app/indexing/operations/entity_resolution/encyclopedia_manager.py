import logging
from typing import List

from app.core.data_model.base import slugify_entity
from app.core.data_model.encyclopedia import EncyclopediaEntry
from app.services.database.encyclopedia_repository import EncyclopediaRepository

logger = logging.getLogger(__name__)

class EncyclopediaManager:
    """
    Acts as the primary service interface for the Encyclopedia system.
    
    This manager bridges the gap between raw entity extraction and the authoritative 
    SQL database. It ensures that entities are correctly normalized and provides 
    unified access to canonical records for resolution and validation purposes.
    """
    
    def __init__(self, repo: EncyclopediaRepository):
        """
        Initializes the manager with the required repository.
        
        Args:
            repo: The repository instance handling persistence and SQL queries.
        """
        self.repo = repo

    async def find_match(self, extracted_title_slug: str, extracted_category: str) -> List[EncyclopediaEntry]:
        """
        Queries the database for entities matching the provided title and category.

        This method handles normalization of the input title and delegates the 
        search criteria (exact match, aliases, and partial containment) to the 
        underlying repository.

        Args:
            extracted_title_slug: The raw or already slugified title to search for.
            extracted_category: The semantic type (e.g., PERSON, EVENT) used to filter candidates.
            
        Returns:
            A list of validated EncyclopediaEntry objects that match the search criteria.
        """
        target_title = slugify_entity(extracted_title_slug)

        return await self.repo.search_by_criteria(
            slug=target_title, 
            category=extracted_category
        )