from abc import ABC, abstractmethod
from PIL import Image
import base64
import io


class BaseLLM(ABC):
    """Abstract base class for LLM wrappers"""
    
    @abstractmethod
    def query(self, query: str, **kwargs) -> str:
        """Abstract method for querying the model"""
        pass
