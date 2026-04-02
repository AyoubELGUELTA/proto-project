from pydantic import BaseModel
from app.models.domain import SiraEntityType

class ExtractionConfig(BaseModel):
    entity_types: list[str] = [e.value for e in SiraEntityType]
    max_gleanings: int = 1

class SummarizationConfig(BaseModel):
    max_summary_length: int = 500  # mots
    max_input_tokens: int = 2000