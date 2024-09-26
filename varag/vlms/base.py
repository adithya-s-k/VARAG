from abc import ABC, abstractmethod
from PIL import Image
import base64
import io

class BaseVLM(ABC):
    @abstractmethod
    def __call__(self, image: Image.Image, query: str) -> str:
        pass

    def _encode_image(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')