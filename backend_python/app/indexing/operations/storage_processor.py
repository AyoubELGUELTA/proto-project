from typing import List
from app.core.data_model.text_units import TextUnit
from PIL import Image
from app.services.storage.base import BaseStorage

class ImageStorageProcessor:
    def __init__(self, storage_service: BaseStorage):
        self.storage = storage_service

    async def process(self, units: List[TextUnit]) -> List[TextUnit]:
        """
        Gère l'upload des images vers le stockage persistant et nettoie les binaires.
        """
        uploaded_cache = {} 

        for unit in units:
            docling_pics = unit.metadata.get("docling_images", [])
            urls = []

            for pic in docling_pics:
                pic_id = id(pic)
                
                # Gestion des doublons en cache (évite l'upload multiple du même logo)
                if pic_id in uploaded_cache:
                    urls.append(uploaded_cache[pic_id])
                    continue

                pil_img = self._extract_pil(pic)
                
                if pil_img is None:
                    continue
                
                # Sécurité : On accepte les images >= 150px (largeur ou hauteur)
                if pil_img.size[0] >= 150 or pil_img.size[1] >= 150:
                    url = self.storage.upload_image(pil_img)
                    if url:
                        urls.append(url)
                        uploaded_cache[pic_id] = url
            
            # Mise à jour des métadonnées finales
            unit.metadata["image_urls"] = urls
            
            # Suppression du binaire Docling pour libérer la RAM
            unit.metadata.pop("docling_images", None) 

        return units
    def _extract_pil(self, item):

        if isinstance(item, Image.Image):
            return item
            
        # Sécurité : si jamais un PictureItem de Docling s'est glissé là par erreur
        if hasattr(item, 'image') and hasattr(item.image, 'pil_image'):
            return item.image.pil_image
            
        return None