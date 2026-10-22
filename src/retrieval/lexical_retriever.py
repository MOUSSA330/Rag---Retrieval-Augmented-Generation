from pathlib import Path

import bm25s

from src.chunking.base import Chunk
from src.retrieval.base import BaseRetriever, ScoredChunk


class LexicalRetriever(BaseRetriever):
    """Search chunks by keyword matching with a BM25 index."""
    def __init__(self, retriever: bm25s.BM25, chunks: list[Chunk]) -> None:
        """Store an already loaded BM25 index and its chunks.

        Args:
            retriever: Loaded bm25s index.
            chunks: Chunks in the same order as when the index was built.
        """
        self.retriever = retriever
        self.chunks = chunks

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        """Return the k chunks with the highest BM25 score for a query.

        The query is tokenized like the corpus was at indexing time
        (same stopwords),
        otherwise matching would silently degrade.

        Args:
            query: The question, in natural language.
            k: Number of results to return, capped at the number of chunks.

        Returns:
            Scored chunks, ordered from the highest score to the lowest.
        """
        query_tokens = bm25s.tokenize([query], stopwords="en")
        # i do [query] because bm25s.tokenize can take kter men query.

        results, scores = self.retriever.retrieve(
            query_tokens, k=min(k, len(self.chunks))
        )

        scored_chunks = []

        for idx, score in zip(results[0], scores[0]):
            new_scored_chunk = ScoredChunk(
                chunk=self.chunks[idx], score=float(score)
            )
            scored_chunks.append(new_scored_chunk)

        # result[0] hitach BM25 ta9der t9aleb 3la bzf dial querys f-
        # d9a wahda  lik q1 -> reses q2 -> reses ....

        # scord maybe can be type of him NumPy !.. so we us float

        return scored_chunks

    @classmethod
    def load(cls, processed_dir: str) -> "LexicalRetriever":
        """Load the BM25 index and the chunks saved by the indexer.

        Args:
            processed_dir: Directory containing bm25_index/ and chunks.pkl.

        Returns:
            A retriever ready to search.

        Raises:
            FileNotFoundError: If the index has not been built yet.
        """
        import pickle

        path = Path(processed_dir)
        index_path = path / "bm25_index"
        chunks_path = path / "chunks.pkl"

        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"Index not found in {processed_dir}. Run `index` first."
            )
        retriever = bm25s.BM25.load(str(index_path))

        with chunks_path.open("rb") as file:
            chunks = pickle.load(file)

        return cls(
            retriever=retriever, chunks=chunks
        )  # == return LexicalRetriever(retriever=retriever, chunks=chunks)
