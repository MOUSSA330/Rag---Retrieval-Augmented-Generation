import json
from pathlib import Path
import sys

import fire
from pydantic import ValidationError

from src.generation.generator import AnswerGenerator
from src.generator_service import GeneratorService
from src.indexer import Indexer
from src.models import (
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.retrieval.lexical_retriever import LexicalRetriever
from src.retriever_service import RetrieverService
from src.validation import validate_k, validate_query

from src.evaluator import evaluate as evaluate_fn
from src.retrieval.semantic_retriever import SemanticRetriever

from src.retrieval.base import BaseRetriever


class RagCLI:
    """
    Command-line commands for indexing, searching,
    answering and evaluating.
    """
    def index(
        self, max_chunk_size: int = 2000, build_semantic: bool = True
    ) -> None:
        """Chunk data/raw and build the lexical index under data/processed.
        Args:
            max_chunk_size: Maximum characters per chunk.
            build_semantic: Also build the semantic (embeddings) index.
        """
        try:
            if (
                not isinstance(max_chunk_size, int) or max_chunk_size <= 0
            ):  # crache here if max_chunk_size not int
                raise ValueError(
                    f"max_chuunk_size be > 0 and must be Valid Integer,"
                    f"got `{max_chunk_size}`"
                )
            indexer = Indexer(
                raw_dir="data/raw",
                processed_dir="data/processed",
                max_chunk_size=max_chunk_size,
            )
            chunks = indexer.run()

            if not isinstance(build_semantic, bool):
                raise ValueError(
                    f"build_semantic be 'True or Flase',"
                    f"got `{build_semantic}`"
                )

            if build_semantic:
                semantic = SemanticRetriever.build(chunks)
                semantic.save("data/processed")
                print("Semantic index built and saved succsefeuly !!.")

        except ValueError as e:
            print(e)
        except FileNotFoundError as error:
            print(f"Error: {error}")
        except OSError as error:
            print(f"Error: could not complete indexing: {error}")
        # except Exception as e:
        #     print(e)

    def search(
        self,
        query: str,
        k: int = 5,
        retrieval_mode: str = "hybrid",
        use_cache: bool = True,
    ) -> None:
        """Print the top-k source locations for a single question.

        Args:
            query: The question to search for.
            k: Number of sources to return.
            retrieval_mode: One of "lexical", "semantic" or "hybrid".
            use_cache: Cache query results in memory.
        """
        try:
            if not isinstance(k, int):
                raise ValueError(f"k be Valid Integer, got `{k}`")
            query = validate_query(query)
            k = validate_k(k)

            retriever = self._build_retriever(retrieval_mode)

            if not isinstance(use_cache, bool):
                raise ValueError(
                    f"use_cache must be True or False, got `{use_cache}`"
                )
            if use_cache:
                from src.caching.cache import CachedRetriever

                retriever = CachedRetriever(retriever)

            service = RetrieverService(retriever=retriever)
            result = service.search(question=query, k=k)

            if not result.retrieved_sources:
                print("No results found.")
                return

            for source in result.retrieved_sources:
                print(
                    f"{source.file_path} "
                    f"[{source.first_character_index}"
                    f":{source.last_character_index}]"
                )
        except (ValueError, FileNotFoundError, Exception) as error:
            print(f"Error: {error}")

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
        retrieval_mode: str = "hybrid",
    ) -> None:
        """
        Run retrieval on every question of a dataset and save the results as
        JSON.

        Args:
            dataset_path: Path to the questions JSON file.
            k: Number of sources to retrieve per question.
            save_directory: Directory where the results file is written.
            retrieval_mode: One of "lexical", "semantic" or "hybrid".
        """

        try:
            k = validate_k(k)
            dataset = self._load_dataset(dataset_path)

            retriever = self._build_retriever(retrieval_mode)
            service = RetrieverService(retriever=retriever)
            results = service.search_dataset(dataset=dataset, k=k)

            output_path = self._save_results(
                results, save_directory, Path(dataset_path).name
            )
            print(f"Saved student_search_results to {output_path}")
        except (
            ValueError,
            FileNotFoundError,
            json.JSONDecodeError,
            ValidationError,
        ) as error:
            print(f"Error: {error}")

    def answer(
        self, query: str, k: int = 5, retrieval_mode: str = "hybrid"
    ) -> None:
        """Retrieve context for one question and print the generated answer.

        Args:
            query: The question to answer.
            k: Number of sources used as context.
            retrieval_mode: One of "lexical", "semantic" or "hybrid".
        """

        try:
            query = validate_query(query)
            k = validate_k(k)

            retriever = self._build_retriever(retrieval_mode)
            retriever_service = RetrieverService(retriever=retriever)
            search_result = retriever_service.search(question=query, k=k)

            generator = AnswerGenerator()
            generator_service = GeneratorService(generator=generator)
            result = generator_service.generate_answer(search_result)

            print(f"\nQuestion: {result.question}")
            print(f"\nAnswer: {result.answer}")
        except (ValueError, FileNotFoundError) as error:
            print(f"Error: {error}")

    def answer_dataset(
        self, student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer"
    ) -> None:
        """Generate answers for saved search results and write them as JSON.

        Args:
            student_search_results_path: Path to the search_dataset output.
            save_directory: Directory where the answers file is written.
        """

        try:
            path = Path(student_search_results_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"File not found: {student_search_results_path}"
                )

            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            student_results = StudentSearchResults.model_validate(data)

            generator = AnswerGenerator()
            generator_service = GeneratorService(generator=generator)
            results = generator_service.generate_dataset(student_results)

            output_path = self._save_results(
                results, save_directory, path.name
            )
            print(f"Saved student_search_results_and_answer to {output_path}")
        except (
            FileNotFoundError,
            json.JSONDecodeError,
            ValidationError,
        ) as error:
            print(f"Error: {error}")

    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        """
        Print recall@k of saved search results against a ground-truth dataset.

        Args:
            student_search_results_path: Path to the search_dataset output.
            dataset_path: Path to the AnsweredQuestions ground-truth file.
        """

        try:
            path = Path(student_search_results_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"File not found: {student_search_results_path}"
                    f" Run search_dataset First!"
                )

            with path.open("r", encoding="utf-8") as file:
                student_data = json.load(file)
            student_results = StudentSearchResults.model_validate(student_data)

            ground_truth = self._load_dataset(dataset_path)

            scores = evaluate_fn(
                student_results=student_results, ground_truth=ground_truth
            )

            print("\nEvaluation Results")
            print("=" * 40)
            for k, score in scores.items():
                print(f"Recall@{k}: {score:.3f} ({score * 100:.1f}%)")
        except (
            FileNotFoundError,
            json.JSONDecodeError,
            ValidationError,
            ValueError,
        ) as error:
            print(f"Error: {error}")

    def index_incremental(self, max_chunk_size: int = 2000) -> None:
        """Re-index only the files that are new or changed since the last run.

        Args:
            max_chunk_size: Maximum characters per chunk.
        """
        try:
            indexer = Indexer(
                raw_dir="data/raw",
                processed_dir="data/processed",
                max_chunk_size=max_chunk_size,
            )
            indexer.run_incremental()
        except FileNotFoundError as error:
            print(f"Error: {error}")
        except OSError as error:
            print(f"Error: could not complete incremental indexing: {error}")

    def serve(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Start the local HTTP API (bonus).

        Args:
            host: Address to listen on.
            port: Port to listen on.
        """
        import uvicorn
        uvicorn.run("src.api.app:app", host=host, port=port)  # redoc / docs

    @staticmethod
    def _build_retriever(mode: str) -> BaseRetriever:
        """Load the retriever matching the requested mode.

        Args:
            mode: One of "lexical", "semantic" or "hybrid".

        Returns:
            A retriever ready to search.

        Raises:
            ValueError: If the mode is unknown.
            FileNotFoundError: If the required index has not been built.
        """

        if mode == "lexical":
            return LexicalRetriever.load(processed_dir="data/processed")

        if mode == "semantic":
            from src.retrieval.semantic_retriever import SemanticRetriever

            return SemanticRetriever.load(processed_dir="data/processed")

        if mode == "hybrid":
            from src.retrieval.hybrid_retriever import HybridRetriever
            from src.retrieval.semantic_retriever import SemanticRetriever

            lexical = LexicalRetriever.load(processed_dir="data/processed")
            semantic = SemanticRetriever.load(processed_dir="data/processed")
            return HybridRetriever(lexical=lexical, semantic=semantic)

        raise ValueError(
            f"Unknown retrieval_mode: {mode}. "
            "Choose from 'lexical', 'semantic', 'hybrid'."
        )

    @staticmethod
    def _load_dataset(dataset_path: str) -> RagDataset:
        """Read a JSON file and validate it as a RagDataset.

        Args:
            dataset_path: Path to the dataset JSON file.

        Returns:
            The validated dataset.

        Raises:
            FileNotFoundError: If the file does not exist.
        """

        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return RagDataset.model_validate(data)

    @staticmethod
    def _save_results(
        results: StudentSearchResults | StudentSearchResultsAndAnswer,
        save_directory: str,
        filename: str,
    ) -> Path:
        """Write results as indented JSON and return the output path.

        Args:
            results: Search results, with or without answers.
            save_directory: Target directory, created if missing.
            filename: Name of the output file.

        Returns:
            Path of the written file.
        """

        output_dir = Path(save_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        with output_path.open("w", encoding="utf-8") as file:
            file.write(results.model_dump_json(indent=2))

        return output_path


def main() -> None:
    """Expose RagCLI through Python Fire, without ever showing a traceback."""

    try:
        fire.Fire(RagCLI)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
