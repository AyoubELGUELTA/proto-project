import asyncio
import pandas as pd
from .summarize_extractor import SummarizeExtractor
from typing import Tuple

async def summarize_descriptions(
    entities_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
    extractor: SummarizeExtractor,
    num_threads: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Applique la summarization sur les entités et les relations en parallèle."""
    
    semaphore = asyncio.Semaphore(num_threads)

    async def throttled_summarize(id, descs):
        async with semaphore:
            return await extractor(id, descs)

    # 1. Résumer les Entités
    entity_tasks = [
        throttled_summarize(row.title, row.description) 
        for row in entities_df.itertuples()
    ]
    summarized_ent_desc = await asyncio.gather(*entity_tasks)
    entities_df["description"] = summarized_ent_desc

    # 2. Résumer les Relations
    rel_tasks = [
        throttled_summarize((row.source, row.target), row.description) 
        for row in relationships_df.itertuples()
    ]
    summarized_rel_desc = await asyncio.gather(*rel_tasks)
    relationships_df["description"] = summarized_rel_desc

    return entities_df, relationships_df