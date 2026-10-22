from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Chunk:
    """A piece of a file with its position in the original content.

    Attributes:
        text: Content of the chunk.
        file_path: Path of the source file, as in the corpus.
        first_character_index: Start offset in the file (inclusive).
        last_character_index: End offset in the file (exclusive).
    """
    text: str
    file_path: str
    first_character_index: int
    last_character_index: int


class BaseChunker(ABC):
    """
    Interface implemented by every chunking strategy.
    """
    @abstractmethod
    def chunk(
        self, content: str, file_path: str, max_chunk_size: int
    ) -> list[Chunk]:
        """Split a file's content into chunks.

        Args:
            content: Raw file content.
            file_path: Path stored in every chunk.
            max_chunk_size: Maximum characters per chunk.

        Returns:
            Chunks ordered by position in the file.
        """
        raise NotImplementedError
