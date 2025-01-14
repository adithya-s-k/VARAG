from abc import ABC, abstractmethod
from typing import List, Callable, Optional
from chonkie import TokenChunker, WordChunker


class BaseChunker(ABC):
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        pass


class FixedTokenChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function

    def split_text(self, text: str) -> List[str]:
        # Split the text into sentences or smaller units        
        chunker = TokenChunker(
            tokenizer="gpt2",  # Supports string identifiers
            chunk_size=self._chunk_size,    # Maximum tokens per chunk
            chunk_overlap=self._chunk_overlap  # Overlap between chunks
        )
        
        return chunker.chunk(text=text)
    
class FixedWordChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function

    def split_text(self, text: str) -> List[str]:
        # Split the text into sentences or smaller units        
        chunker = WordChunker(
            tokenizer="gpt2",  # Supports string identifiers
            chunk_size=self._chunk_size,    # Maximum tokens per chunk
            chunk_overlap=self._chunk_overlap  # Overlap between chunks
        )
        
        return chunker.chunk(text=text)
    
