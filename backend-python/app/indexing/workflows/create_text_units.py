
from typing import List
from app.indexing.operations.pdf.loader import DocumentLoader
from app.indexing.operations.pdf.chunker import DocumentChunker
from app.indexing.operations.pdf.spatial_processor import SpatialProcessor
from app.indexing.operations.text.metadata_refiner import MetadataRefiner
from app.indexing.operations.text.text_splitter import TextSplitter
from app.indexing.operations.storage_processor import ImageStorageProcessor 
from app.services.storage.minio_storage import MinioStorage
from app.models.domain import TextUnit

async def workflow_create_text_units(file_path: str, identity_text: str = None) -> List[TextUnit]:
    """
    Orchestre le passage d'un PDF brut à une liste de TextUnits raffinées et persistées.
    """
    # 1. Chargement et conversion
    loader = DocumentLoader()
    converter = loader.get_converter(file_path)
    result = converter.convert(file_path)
    doc = result.document

    # 2. Premier découpage (Layout-Aware)
    chunker = DocumentChunker()
    dl_chunks = chunker.chunk(doc)

    # 3. Enrichissement Spatial (Images & Tables)
    spatial_proc = SpatialProcessor()
    units = spatial_proc.enrich_with_spatial_data(doc, dl_chunks)

    # 4. Raffinement des métadonnées (Titres & Héritage)
    valid_titles = MetadataRefiner.extract_titles_from_identity(identity_text)
    refiner = MetadataRefiner(valid_titles=valid_titles)
    refined_units = refiner.refine_units(units)

    # 5. Sub-splitting final (Gestion des limites de tokens)
    splitter = TextSplitter(max_tokens=800, overlap=100)
    final_units = splitter.split_units(refined_units)

         
    # 6. Persistance et nettoyage (MinIO)
    storage_service = MinioStorage()
    storage_proc = ImageStorageProcessor(storage_service)
    
    final_units = await storage_proc.process(final_units)

    return final_units
    