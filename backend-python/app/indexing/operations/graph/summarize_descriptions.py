import asyncio
import pandas as pd
from typing import Tuple
from .summarize_extractor import SummarizeExtractor

async def summarize_descriptions(
    entities_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
    extractor: SummarizeExtractor,
    num_threads: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orchestre la summarization massive via un sémaphore pour respecter les Rate Limits.
    """
    semaphore = asyncio.Semaphore(num_threads)

    async def throttled_summarize(item_id, descs):
        async with semaphore:
            return await extractor(item_id, descs)

    # 1. Traitement des Entités
    if not entities_df.empty:
        # On ne résume que si nécessaire (ex: > 1 desc) pour économiser des tokens
        tasks = [
            throttled_summarize(row.title, row.description) 
            for row in entities_df.itertuples()
        ]
        entities_df["description"] = await asyncio.gather(*tasks)

    # 2. Traitement des Relations
    if not relationships_df.empty:
        tasks = [
            throttled_summarize(f"{row.source} -> {row.target}", row.description) 
            for row in relationships_df.itertuples()
        ]
        relationships_df["description"] = await asyncio.gather(*tasks)

    return entities_df, relationships_df