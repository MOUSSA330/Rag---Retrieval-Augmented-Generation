import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.chunking.base import Chunk
from src.retrieval.base import BaseRetriever, ScoredChunk

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticRetriever(BaseRetriever):
    """
    Search chunks by meaning, using cosine similarity between embeddings.
    """
    def __init__(
        self,
        model: SentenceTransformer,  # encode query -> query embedding
        embeddings: np.ndarray,
        chunks: list[Chunk],
    ) -> None:
        """Store the embedding model, the chunk embeddings and the chunks.

        Args:
            model: Sentence-transformers model used to embed queries.
            embeddings: Matrix of shape (number of chunks, embedding size).
            chunks: Chunks in the same order as the rows of embeddings.
        """

        self.model = model
        self.embeddings = embeddings  # Sentence embeddings heatmap
        self.chunks = chunks

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        """Return the k chunks whose embeddings are closest to the query.

        Embeddings are L2-normalized, so the dot product equals
        the cosine similarity.

        Args:
            query: The question, in natural language.
            k: Number of results to return, capped at the number of chunks.

        Returns:
            Scored chunks, ordered from the highest similarity to the lowest.
        """
        query_embedding = self.model.encode(query, normalize_embeddings=True)

        # @ ->> MAtrix Multiplication
        scores = self.embeddings @ query_embedding

        top_k = min(k, len(self.chunks))
        # here we take best results
        # np.argsorst sourted scores low to up and return indeces of him
        # [::-1] change tartib bach ywali men lakbir l sghir [:top] topk
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            ScoredChunk(chunk=self.chunks[idx], score=float(scores[idx]))
            for idx in top_indices
        ]

    @classmethod
    def build(
        cls, chunks: list[Chunk], model_name: str = DEFAULT_MODEL_NAME
    ) -> "SemanticRetriever":
        """Embed every chunk and build a semantic index.

        Args:
            chunks: Chunks to embed.
            model_name: Sentence-transformers model name.

        Returns:
            A retriever ready to search.
        """

        model = SentenceTransformer(model_name)  # we can do text -> embedding
        texts = [chunk.text for chunk in chunks]
        embeddings = model.encode(
            texts, normalize_embeddings=True, show_progress_bar=True
        )
        return cls(model=model, embeddings=embeddings, chunks=chunks)

    def save(self, processed_dir: str) -> None:
        """Save the embeddings and the chunks under processed_dir.

        Args:
            processed_dir: Directory where semantic_embeddings.npy and
                semantic_chunks.pkl are written.
        """
        path = Path(processed_dir)
        path.mkdir(parents=True, exist_ok=True)

        np.save(path / "semantic_embeddings.npy", self.embeddings)
        with (path / "semantic_chunks.pkl").open("wb") as file:
            pickle.dump(self.chunks, file)

    @classmethod
    def load(
        cls, processed_dir: str, model_name: str = DEFAULT_MODEL_NAME
    ) -> "SemanticRetriever":
        """Load the semantic index saved by save().

        The model must be the same one used to build the index,
        otherwise query and
        chunk embeddings are not comparable.

        Args:
            processed_dir: Directory containing the saved index files.
            model_name: Sentence-transformers model name.

        Returns:
            A retriever ready to search.

        Raises:
            FileNotFoundError: If the semantic index has not been built yet.
        """

        path = Path(processed_dir)
        embeddings_path = path / "semantic_embeddings.npy"
        chunks_path = path / "semantic_chunks.pkl"

        if not embeddings_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"Semantic index not found in {processed_dir}"
                f"Please run `index` and save him in {processed_dir}."
            )

        model = SentenceTransformer(model_name)
        embeddings = np.load(embeddings_path)

        with chunks_path.open("rb") as file:
            chunks = pickle.load(file)

        return cls(model=model, embeddings=embeddings, chunks=chunks)
