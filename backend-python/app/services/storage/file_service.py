import os
import logging
from fastapi import UploadFile
from app.core.settings import settings

logger = logging.getLogger(__name__)

class FileService:
    def __init__(self):
        self.base_path = getattr(settings, "local_storage_path", "data/storage") #TODO TOSEE 
        os.makedirs(self.base_path, exist_ok=True)

    async def save_uploaded_file(self, upload_file: UploadFile, custom_name: str) -> str:
        # On s'assure que l'extension est correcte
        ext = ".pdf" if not upload_file.filename.endswith(".pdf") else ""
        file_path = os.path.join(self.base_path, f"{custom_name}{ext}")
        
        try:
            # Rembobiner le curseur au cas où le fichier a déjà été lu
            await upload_file.seek(0)
            
            with open(file_path, "wb") as buffer:
                while content := await upload_file.read(1024 * 1024):
                    buffer.write(content)
            
            logger.info(f"💾 Fichier sauvegardé localement : {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'écriture disque : {e}")
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e