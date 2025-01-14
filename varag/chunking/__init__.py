from abc import ABC, abstractmethod
from typing import List, Callable, Optional
from chonkie import TokenChunker, WordChunker,SentenceChunker


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
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

    def chunk(self, text: str) -> List[str]:
        # Split the text into sentences or smaller units        
        chunker = TokenChunker(
            tokenizer="gpt2",  # Supports string identifiers
            chunk_size=self._chunk_size,    # Maximum tokens per chunk
            chunk_overlap=self._chunk_overlap  # Overlap between chunks
        )
        
        chunks = chunker.chunk(text=text)
        return [chunk.text for chunk in chunks]
    
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

    def chunk(self, text: str) -> List[str]:
        # Split the text into sentences or smaller units        
        chunker = WordChunker(
            tokenizer="gpt2",  # Supports string identifiers
            chunk_size=self._chunk_size,    # Maximum tokens per chunk
            chunk_overlap=self._chunk_overlap  # Overlap between chunks
        )
        
        chunks = chunker.chunk(text=text)
        return [chunk.text for chunk in chunks]
    
class FixedSentenceChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_sentences_per_chunk: int = 1,
        length_function: Callable[[str], int] = len,
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_sentences_per_chunk = min_sentences_per_chunk
        self._length_function = length_function

    def chunk(self, text: str) -> List[str]:
        # Basic initialization with default parameters
        chunker = SentenceChunker(
            tokenizer="gpt2",                # Supports string identifiers
            chunk_size=self._chunk_size,                  # Maximum tokens per chunk
            chunk_overlap=self._chunk_overlap,               # Overlap between chunks
            min_sentences_per_chunk=self._min_sentences_per_chunk       # Minimum sentences in each chunk
        )
        
        chunks = chunker.chunk(text=text)
        return [chunk.text for chunk in chunks]
