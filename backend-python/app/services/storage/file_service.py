import os
import logging
from fastapi import UploadFile
from app.core.settings import settings

logger = logging.getLogger(__name__)

class FileService:
    """
    Service responsible for handling local file persistence during the ingestion process.
    
    It acts as a 'Staging Area', saving files uploaded via the API to the local 
    file system so they can be processed by downstream tools like Docling or PDF parsers.
    """

    def __init__(self):
        # Fallback to 'data/storage' if the setting is not explicitly defined
        self.base_path = getattr(settings, "local_storage_path", "data/storage")
        os.makedirs(self.base_path, exist_ok=True)
        logger.debug(f"📁 Local storage initialized at: {self.base_path}")

    async def save_uploaded_file(self, upload_file: UploadFile, custom_name: str) -> str:
        """
        Saves an uploaded FastAPI file to the local storage path.
        
        Args:
            upload_file (UploadFile): The raw file received from the client.
            custom_name (str): The desired filename (usually the document ID).
            
        Returns:
            str: The absolute path to the saved file.
            
        Raises:
            Exception: If the disk write fails, the partial file is cleaned up.
        """
        # Ensure the filename has a .pdf extension for the loaders
        extension = ".pdf" if not upload_file.filename.lower().endswith(".pdf") else ""
        file_path = os.path.join(self.base_path, f"{custom_name}{extension}")
        
        logger.info(f"💾 Saving uploaded file: {upload_file.filename} -> {file_path}")
        
        try:
            # Ensure the cursor is at the beginning of the file (important for re-reads)
            await upload_file.seek(0)
            
            bytes_written = 0
            with open(file_path, "wb") as buffer:
                # Read in 1MB chunks to keep memory usage low
                while content := await upload_file.read(1024 * 1024):
                    buffer.write(content)
                    bytes_written += len(content)
            
            logger.info(f"✅ File successfully staged: {file_path} ({bytes_written // 1024} KB written).")
            return file_path
            
        except Exception as e:
            logger.error(f"❌ Disk write error for file {custom_name}: {e}")
            # Clean up the partial file to avoid corruption/ghost files
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"🧹 Cleaned up partial file: {file_path}")
            raise e