from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.chunking.base import Chunk


@dataclass
class ScoredChunk:
    """A retrieved chunk together with its relevance score.

    Attributes:
        chunk: The retrieved chunk.
        score: Relevance to the query; higher means more relevant.
    """
    chunk: Chunk
    score: float


class BaseRetriever(ABC):
    """
    Interface implemented by every retrieval strategy.
    """
    @abstractmethod
    def search(self, query: str, k: int) -> list[ScoredChunk]:
        """Return the k chunks most relevant to a query.

        Args:
            query: The question, in natural language.
            k: Number of results to return.

        Returns:
            Scored chunks, ordered from the highest score to the lowest.
        """
        raise NotImplementedError
