from .base import get_connection, release_connection 
from .init_db import init_db, seed_system_tags
from .documents import get_documents, get_or_create_document
from .chunks import (
    store_chunks_batch, 
    store_identity_chunk, 
    fetch_identities_by_doc_ids,
    get_chunk_with_metadata,
    update_chunks_with_ai_data
)
from .entities import (
    link_entity_to_chunk, 
    resolve_entity,
    finalize_entity_graph,
    compute_cooccurrences_for_document,
    generate_global_summaries_for_document,
    get_top_cooccurrences
)

# Cela permet de faire : from db import init_db, finalize_entity_graph...

# Cela permet de faire : from db import init_db, store_chunks_batch...