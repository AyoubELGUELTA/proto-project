import os
from unstructured.partition.pdf import partition_pdf


def partition_document(file_path: str):
    """Extract elements from PDF using unstructured"""
    
    extraction_strategy = os.getenv("PDF_EXTRACTION_STRATEGY", "hi_res")

    try:
        elements = partition_pdf(
            filename=file_path,
            strategy=extraction_strategy,
            infer_table_structure=True,
            languages=["fra", "eng"],
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True
        )
        return elements
    except Exception as e:
        print(f"❌ Erreur lors du partitionnement du PDF: {e}")
        raise



