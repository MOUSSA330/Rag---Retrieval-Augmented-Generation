from typing import List
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from src.chunking.base import BaseChunker, Chunk


class TextChunker(BaseChunker):
    """
    Split Markdown at headers first, then at paragraphs and lines.
    """
    def chunk(
        self, content: str, file_path: str, max_chunk_size: int
    ) -> List[Chunk]:
        """
        Split Markdown content into chunks of at most max_chunk_size
        characters.

        Args:
            content: Raw content of the .md file.
            file_path: Path stored in every chunk.
            max_chunk_size: Maximum characters per chunk.

        Returns:
            Chunks ordered by position in the file.
        """

        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,  # enum members
            chunk_size=max_chunk_size,
            chunk_overlap=200,
            add_start_index=True,
        )

        documents = splitter.create_documents([content])

        chunks: List[Chunk] = []

        for doc in documents:
            start = doc.metadata["start_index"]
            end = start + len(doc.page_content)  # exclusive index
            chunks.append(
                Chunk(
                    text=doc.page_content,
                    file_path=file_path,
                    first_character_index=start,
                    last_character_index=end,
                )
            )
        return chunks


# from_language is return type -> RecursiveCharacterTextSplitter.
# Class -> Function -> Blocks -> Lines -> Characters
