from tqdm import tqdm

from src.models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    UnansweredQuestion,
)
from src.retrieval.base import BaseRetriever, ScoredChunk


class RetrieverService:
    """
    Run searches with any BaseRetriever and build the output models.
    """

    def __init__(self, retriever: BaseRetriever) -> None:
        """Store the retriever, whatever its strategy (lexical,
        semantic, hybrid).

        Args:
            retriever: Retriever used for every search.
        """
        self.retriever = retriever

    @staticmethod
    def _to_minimal_source(scored_chunk: ScoredChunk) -> MinimalSource:
        """Convert an internal scored chunk into an output source,
        dropping the score.

        Args:
            scored_chunk: Result returned by a retriever.

        Returns:
            The source location of the chunk.
        """
        chunk = scored_chunk.chunk
        return MinimalSource(
            file_path=chunk.file_path,
            first_character_index=chunk.first_character_index,
            last_character_index=chunk.last_character_index,
        )

    def search(
        self, question: str, k: int, question_id: str | None = None
    ) -> MinimalSearchResults:
        """Search one question and return its retrieved sources.

        Args:
            question: The question to search for.
            k: Number of sources to retrieve.
            question_id: Id to keep; a new UUID is generated if None.

        Returns:
            The question with its ranked sources.
        """
        if question_id is None:
            question_id = UnansweredQuestion(question=question).question_id

        scored_chunks = self.retriever.search(query=question, k=k)

        retrieved_sources = [
            self._to_minimal_source(sc) for sc in scored_chunks
        ]

        return MinimalSearchResults(
            question_id=question_id,
            question=question,
            retrieved_sources=retrieved_sources,
        )

    def search_dataset(
        self, dataset: RagDataset, k: int
    ) -> StudentSearchResults:
        """Search every question of a dataset.

        Args:
            dataset: Questions to search, keeping their question_id.
            k: Number of sources to retrieve per question.

        Returns:
            Results for all questions, and the k used.
        """

        search_results = [
            self.search(question=q.question, k=k, question_id=q.question_id)
            for q in tqdm(dataset.rag_questions, desc="Searching")
        ]

        return StudentSearchResults(search_results=search_results, k=k)
