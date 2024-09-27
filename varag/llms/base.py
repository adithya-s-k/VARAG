from abc import ABC, abstractmethod
from PIL import Image
import base64
import io


class BaseLLM(ABC):
    @abstractmethod
    def __call__(self, system_prompt: str, context: str, query: str) -> str:
        pass
