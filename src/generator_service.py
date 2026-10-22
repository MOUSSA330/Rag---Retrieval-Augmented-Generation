from tqdm import tqdm

from src.generation.generator import AnswerGenerator
from src.models import (
    MinimalAnswer,
    MinimalSearchResults,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)


class GeneratorService:
    """
    Turn search results into answers using an AnswerGenerator.
    """
    def __init__(self, generator: AnswerGenerator) -> None:
        """Store the generator, loaded once and reused for every question.

        Args:
            generator: Loaded AnswerGenerator (Qwen3-0.6B).
        """
        self.generator = generator

    def generate_answer(
        self, search_result: MinimalSearchResults
    ) -> MinimalAnswer:
        """Generate an answer for one search result.

        Args:
            search_result: Question with its retrieved sources.

        Returns:
            The same search result completed with the generated answer.
        """
        answer = self.generator.generate(
            question=search_result.question,
            sources=search_result.retrieved_sources,
        )

        return MinimalAnswer(
            question_id=search_result.question_id,
            question=search_result.question,
            retrieved_sources=search_result.retrieved_sources,
            answer=answer,
        )

    def generate_dataset(
        self, student_results: StudentSearchResults
    ) -> StudentSearchResultsAndAnswer:
        """Generate answers for every question of a search results file.

        Args:
            student_results: Output of search_dataset.

        Returns:
            Search results with an answer for each question, and the same k.
        """
        answers = [
            self.generate_answer(result)
            for result in tqdm(
                student_results.search_results, desc="Generating answers"
            )
        ]

        return StudentSearchResultsAndAnswer(
            search_results=answers, k=student_results.k
        )
