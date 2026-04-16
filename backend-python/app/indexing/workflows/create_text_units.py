import logging
from typing import List, Optional
from app.indexing.operations.pdf.loader import DocumentLoader
from app.indexing.operations.pdf.chunker import DocumentChunker
from app.indexing.operations.pdf.spatial_processor import SpatialProcessor
from app.indexing.operations.text.metadata_refiner import MetadataRefiner
from app.indexing.operations.text.text_splitter import TextSplitter
from app.indexing.operations.storage_processor import ImageStorageProcessor 
from app.services.storage.minio_storage import MinioStorage
from app.models.domain import TextUnit

logger = logging.getLogger(__name__)

async def workflow_create_text_units(file_path: str, identity_text: Optional[str] = None) -> List[TextUnit]:
    """
    Orchestrates the transformation of a raw PDF into refined, persisted TextUnits.
    
    The workflow follows these stages:
    1. Technical Conversion: PDF to structured Document (Docling).
    2. Layout-Aware Chunking: Initial breakdown respecting document geometry.
    3. Spatial Enrichment: Anchoring images/tables to text via BBox analysis.
    4. Metadata Refinement: Cleaning titles and propagating contextual headings.
    5. Token Safety Splitting: Ensuring units fit within embedding context windows.
    6. Asset Persistence: Offloading Base64 images to MinIO storage.
    
    Args:
        file_path: Path to the source PDF file.
        identity_text: Optional OUTLINE from the Identity Card for heading validation.
        
    Returns:
        List[TextUnit]: A list of ready-to-index text units with persistent image URLs.
    """
    logger.info(f"🚀 Starting TextUnit creation workflow for: {file_path}")

    try:
        # 1. Technical Loading & Conversion
        loader = DocumentLoader()
        converter = loader.get_converter(file_path)
        
        logger.info("⏳ Converting PDF to structured format (Docling)...")
        result = converter.convert(file_path)
        doc = result.document
        logger.info("✅ Technical conversion successful.")

        # 2. Layout-Aware Chunking
        chunker = DocumentChunker()
        dl_chunks = chunker.chunk(doc)
        if not dl_chunks:
            logger.error(f"❌ DocumentChunker returned 0 chunks for {file_path}. Aborting.")
            return []

        # 3. Spatial Enrichment (Images & Tables)
        spatial_proc = SpatialProcessor()
        units = spatial_proc.enrich_with_spatial_data(doc, dl_chunks)
        logger.info(f"📍 Spatial enrichment completed: {len(units)} units created.")

        # 4. Metadata Refinement (Headings & Heritage)
        valid_titles = MetadataRefiner.extract_titles_from_identity(identity_text) if identity_text else []
        refiner = MetadataRefiner(valid_titles=valid_titles)
        refined_units = refiner.refine_units(units)

        # 5. Final Sub-splitting (Token limits management)
        # We use a tighter limit (800) here to be safe for diverse embedding models
        
        splitter = TextSplitter(max_tokens=800, overlap=120)
        final_units = splitter.split_units(refined_units)

        # 6. Persistence & Image Storage (MinIO)
        # Offloads heavy Base64 strings to cloud storage to keep the Graph database light

        logger.info("☁️ Persisting visual assets to MinIO...")
        storage_service = MinioStorage()
        storage_proc = ImageStorageProcessor(storage_service)
        
        final_units = await storage_proc.process(final_units)

        logger.info(f"✨ Workflow finished: {len(final_units)} TextUnits ready for Graph extraction.")
        return final_units

    except Exception as e:
        logger.critical(f"💥 Critical failure in workflow_create_text_units: {e}", exc_info=True)
        return []