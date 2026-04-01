from abc import ABC, abstractmethod
from PIL import Image

class BaseStorage(ABC):
    @abstractmethod
    def upload_image(self, pil_image: Image.Image) -> str:
        """Upload et retourne l'URL publique."""
        pass