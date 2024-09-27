from abc import ABC, abstractmethod
from typing import List, Callable, Optional


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
        splits = text.split(". ")
        return self._merge_splits(splits, ". ")

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        docs = []
        current_doc = []
        total = 0
        for d in splits:
            _len = self._length_function(d)
            if _len > self._chunk_size:
                # If a single split is longer than chunk_size, we need to handle it separately
                if current_doc:
                    docs.append(self._join_docs(current_doc, separator))
                    current_doc = []
                    total = 0
                # Split the long sentence into smaller parts
                sub_splits = self._split_long_sentence(d, separator)
                docs.extend(sub_splits)
            elif total + _len > self._chunk_size:
                if current_doc:
                    docs.append(self._join_docs(current_doc, separator))
                current_doc = [d]
                total = _len
            else:
                current_doc.append(d)
                total += _len
        if current_doc:
            docs.append(self._join_docs(current_doc, separator))
        return [doc for doc in docs if doc is not None]

    def _split_long_sentence(self, sentence: str, separator: str) -> List[str]:
        words = sentence.split()
        chunks = []
        current_chunk = []
        current_length = 0
        for word in words:
            word_len = self._length_function(word)
            if current_length + word_len > self._chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_len
            else:
                current_chunk.append(word)
                current_length += word_len
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        text = text.strip()
        if text == "":
            return None
        else:
            return text
