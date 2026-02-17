from .init_db import init_db
from .documents import get_documents, get_or_create_document
from .chunks import (
    store_chunks_batch, 
    store_identity_chunk, 
    fetch_identities_by_doc_ids,
    get_chunk_with_metadata,
    update_chunks_with_ai_data
)
from .entities import link_entity_to_chunk, resolve_entity

# Cela permet de faire : from db import init_db, store_chunks_batch...