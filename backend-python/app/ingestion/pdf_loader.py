import os
from unstructured.partition.pdf import partition_pdf
import base64


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


MAX_IMAGE_MB = 5
MAX_IMAGE_PIXELS = 10_000 * 10_000  # garde-fou absurde

def filter_image_elements(elements):
    filtered = []
    print ("voici les elements : ", elements)

    for el in elements:
        # --- On garde tout sauf les images problématiques ---
        if el.category != "Image":
            filtered.append(el)
            continue

        meta = el.metadata

        # --- Sécurité ---
        if meta is None or not hasattr(meta, "image_base64"):
            print("⚠️ Image sans payload base64, ignorée")
            continue

        b64 = meta.image_base64
        if not b64:
            continue

        # --- Taille ---
        try:
            size_bytes = len(base64.b64decode(b64))
        except Exception:
            print("⚠️ Base64 invalide, image ignorée")
            continue

        size_mb = size_bytes / (1024 * 1024)

        if size_mb > MAX_IMAGE_MB:
            print(f"⚠️ Image ignorée (trop lourde): {size_mb:.2f} MB")
            continue

        # --- Dimensions (optionnel) ---
        w = getattr(meta, "image_width", None)
        h = getattr(meta, "image_height", None)

        if w and h and (w * h) > MAX_IMAGE_PIXELS:
            print(f"⚠️ Image ignorée (dimensions excessives): {w}x{h}")
            continue

        filtered.append(el)

    return filtered