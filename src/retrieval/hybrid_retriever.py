from src.retrieval.base import BaseRetriever, ScoredChunk


class HybridRetriever(BaseRetriever):
    """
    Merge a lexical and a semantic retriever into a single ranking.
    """
    def __init__(
        self,
        lexical: BaseRetriever,
        semantic: BaseRetriever,
        alpha: float = 0.5,
    ) -> None:
        """Store the two retrievers and the weight given to the lexical one.

        Args:
            lexical: Lexical retriever (BM25).
            semantic: Semantic retriever (embeddings).
            alpha: Weight of the lexical score, between 0 (semantic only) and
                1 (lexical only).
        """

        self.lexical = lexical
        self.semantic = semantic
        self.alpha = alpha  # chal na3ti l lexical mo9aratan m3a semantic

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        """Return the k best chunks according to the combined score.

        Each retriever returns its top 2*k results, whose scores are min-max
        normalized and then merged as
        alpha * lexical + (1 - alpha) * semantic. A chunk found by only one
        retriever gets 0 for the other.

        Args:
            query: The question, in natural language.
            k: Number of results to return.

        Returns:
            Scored chunks, ordered from the highest combined
            score to the lowest.
        """
        lexical_results = self.lexical.search(query, k=k * 2)
        semantic_results = self.semantic.search(query, k=k * 2)

        raw_scores = {}
        for r in lexical_results:
            key = self._key(r)
            raw_scores[key] = r.score
        # {(path, first, last) : numner -> score for this chunks}

        lexical_scores = self._normalize(raw_scores)

        raw_scores = {}
        for r in semantic_results:
            key = self._key(r)
            raw_scores[key] = r.score

        semantic_scores = self._normalize(raw_scores)

        all_keys = set(lexical_scores) | set(semantic_scores)  # | like union

        chunk_by_key = {}
        for r in lexical_results + semantic_results:
            key = self._key(r)
            chunk_by_key[key] = r.chunk

        combined = []

        for key in all_keys:
            chunk = chunk_by_key[key]

            lex_score = lexical_scores.get(key, 0.0)
            sem_score = semantic_scores.get(key, 0.0)

            final_score = (self.alpha * lex_score) + (
                (1 - self.alpha) * sem_score
            )

            scored_chunk = ScoredChunk(chunk=chunk, score=final_score)
            combined.append(scored_chunk)

        combined.sort(key=lambda sc: sc.score, reverse=True)

        return combined[:k]

    # `_key` for check if semantic and lexical are have same chunks.
    @staticmethod
    def _key(scored_chunk: ScoredChunk) -> tuple[str, int, int]:
        """Identify a chunk by its file and character range.

        The two retrievers hold separate copies of the chunks,
        so this key is what
        recognizes the same chunk in both result lists.

        Args:
            scored_chunk: Result returned by a retriever.

        Returns:
            Tuple of file path, first and last character index.
        """
        chunk = scored_chunk.chunk
        return (
            chunk.file_path,
            chunk.first_character_index,
            chunk.last_character_index,
        )

    @staticmethod
    def _normalize(scores: dict) -> dict:
        """Rescale scores to [0, 1] with min-max scaling.

        BM25 and cosine scores are on different scales, so they must
        be normalized
        before being combined.

        Args:
            scores: Raw scores indexed by chunk key.

        Returns:
            The same keys with scores between 0 and 1. If all scores are equal,
            each one becomes 1.0.
        """
        if not scores:
            return {}

        values = list(scores.values())
        min_score = min(values)
        max_score = max(values)

        normalized_scores = {}

        if max_score == min_score:
            for key in scores:
                normalized_scores[key] = 1.0
            return normalized_scores

        for key, score in scores.items():
            normalized_scores[key] = (score - min_score) / (
                max_score - min_score
            )

        return normalized_scores

    #  a - min_a / max_a - min_a -> scall btw 0 to 1
