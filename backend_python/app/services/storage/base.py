from abc import ABC, abstractmethod
from PIL import Image

class BaseStorage(ABC):
    """
    Abstract Base Class for multimedia storage services.
    
    This interface defines the contract for any storage provider (MinIO, S3, GCS, Local).
    It ensures that the rest of the application remains decoupled from the 
    specific storage implementation.
    """

    @abstractmethod
    def upload_image(self, pil_image: Image.Image) -> str:
        """
        Processes and uploads a PIL image to a persistent storage provider.
        
        Args:
            pil_image (Image.Image): The PIL Image object to be stored.
            
        Returns:
            str: The public URL or unique identifier of the stored asset. 
                 Returns an empty string if the upload fails.
        """
        pass