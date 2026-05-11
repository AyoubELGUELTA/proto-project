# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# THE LOGIC OF THIS FILE WAS MOVED TO CORE/CONFIG/ FOLDER, IT REMAINS FOR TESTS PURPOSES ONLY

from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import List, Optional

@dataclass
class ExtractGraphPrompts:
    """Templates de prompts pour l'extraction de graphe (Matches MS)."""
    extraction_prompt: str
    continue_prompt: str
    loop_prompt: str

class ExtractGraphConfig(BaseModel):
    """Configuration calquée sur GraphRAG pour l'extraction d'entités."""
    
    # Identifiants du modèle
    model_id: str = Field(default="gpt-4o-mini")
    model_instance_name: Optional[str] = None
    
    # Paramètres d'extraction
    prompt: Optional[str] = None  # Chemin vers un fichier .txt ou None
    entity_types: List[str] = Field(default=["PERSON", "ORGANIZATION", "LOCATION", "EVENT"])
    max_gleanings: int = Field(default=1, ge=0)
    
    # Délimiteurs de format (Strict MS)
    tuple_delimiter: str = "<|>"
    record_delimiter: str = "##"
    completion_delimiter: str = "<|COMPLETE|>"

class SummarizeDescriptionsConfig(BaseModel):
    """Configuration pour le merging des descriptions (Matches MS)."""
    model_id: str = "gpt-4o-mini"
    max_tokens: int = 500
    temperature: float = 0.0

class GlobalIndexingConfig(BaseModel):
    """L'objet racine qui contient toutes les sections de config."""
    extract_graph: ExtractGraphConfig = Field(default_factory=ExtractGraphConfig)
    summarize_descriptions: SummarizeDescriptionsConfig = Field(default_factory=SummarizeDescriptionsConfig)
    
    # Paramètres de stockage/cache
    cache_enabled: bool = True
    storage_base_dir: str = "data/artifacts/graph"

class SummarizeDescriptionsConfig(BaseModel):
    max_summary_length: int = 500
    max_input_tokens: int = 2000
    num_threads: int = 4
