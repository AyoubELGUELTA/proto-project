from pydantic import BaseModel
from app.models.domain import SiraEntityType


class ExtractionConfig(BaseModel):
    entity_types: list[str] = [e.value for e in SiraEntityType]
    max_gleanings: int = 1
    record_delimiter: str = "##" 


class SummarizationConfig(BaseModel):
    max_summary_length: int = 350  # words
    max_input_tokens: int = 2000
    entity_batch_size: int = 10


class EntityResolvingConfig(BaseModel):
    max_cluster_batch: int = 22


extraction_config = ExtractionConfig()
summarization_config = SummarizationConfig()
entity_resolving_config = EntityResolvingConfig()

# Extraction
ENTITY_TYPES = extraction_config.entity_types
MAX_GLEANINGS = extraction_config.max_gleanings
RECORD_DELIMITER = extraction_config.record_delimiter

# Summarization
MAX_SUMMARY_LENGTH = summarization_config.max_summary_length
MAX_INPUT_TOKENS = summarization_config.max_input_tokens
ENTITY_BATCH_SIZE = summarization_config.entity_batch_size

# Entity Resolving
MAX_CLUSTER_BATCH = entity_resolving_config.max_cluster_batch
