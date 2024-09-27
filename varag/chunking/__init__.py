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
            if total + _len > self._chunk_size:
                if total > self._chunk_size:
                    print(
                        f"Created a chunk of size {total}, which is longer than the specified {self._chunk_size}"
                    )
                if current_doc:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    while total > self._chunk_overlap or (
                        total + _len > self._chunk_size and total > 0
                    ):
                        total -= self._length_function(current_doc[0])
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        text = text.strip()
        if text == "":
            return None
        else:
            return text
