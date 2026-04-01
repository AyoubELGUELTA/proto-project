import os
import shutil
from fastapi import UploadFile
import logging

class FileService:
    def __init__(self, base_path: str = "data/storage"):
        self.base_path = base_path
        # On s'assure que le dossier de stockage existe dès le départ
        os.makedirs(self.base_path, exist_ok=True)

    async def save_uploaded_file(self, upload_file: UploadFile, custom_name: str) -> str:
        """
        Sauvegarde un UploadFile FastAPI sur le disque de manière efficace.
        """
        file_path = os.path.join(self.base_path, f"{custom_name}.pdf")
        
        try:
            # On utilise un "buffer" pour ne pas charger tout le fichier en RAM
            with open(file_path, "wb") as buffer:
                # On lit par morceaux de 1Mo (1024 * 1024)
                while content := await upload_file.read(1024 * 1024):
                    buffer.write(content)
            
            logging.info(f"💾 Fichier sauvegardé localement : {file_path}")
            return file_path
            
        except Exception as e:
            logging.error(f"❌ Erreur lors de l'écriture disque : {e}")
            # Si le fichier a été partiellement écrit, on tente de le supprimer
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e # On relance pour que le IngestionContext attrape l'erreur