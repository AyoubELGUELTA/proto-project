import logging
from typing import List, Dict, Any, Tuple
import pandas as pd
from app.services.graph.graph_service import GraphService

logger = logging.getLogger(__name__)

async def extract_graph(
    text_units: List[Any],
    graph_service: GraphService,
    domain_context: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Workflow orchestrateur. Il reçoit les text_units et délègue l'extraction 
    et le premier nettoyage au GraphService.
    """
    logger.info(f"Starting graph extraction for {len(text_units)} units.")
    
    # On utilise la méthode run_pipeline du service (Parallelism + Summarization)
    entities_df, relationships_df = await graph_service.run_pipeline(
        text_units=text_units,
        domain_context=domain_context
    )
    
    return entities_df, relationships_df