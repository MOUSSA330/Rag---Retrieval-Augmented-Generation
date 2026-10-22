from functools import lru_cache

from src.retrieval.base import BaseRetriever, ScoredChunk
from src.retrieval.lexical_retriever import LexicalRetriever


@lru_cache(maxsize=1)
def get_cached_lexical_retriever(processed_dir: str) -> LexicalRetriever:
    """Load the lexical retriever once and reuse it on later calls.

    Args:
        processed_dir: Directory containing the saved index.

    Returns:
        The same LexicalRetriever instance for a given processed_dir.
    """
    return LexicalRetriever.load(processed_dir=processed_dir)


# add new fuature its cache.


class CachedRetriever(BaseRetriever):
    """
    Wrap any retriever and cache its results per (query, k).
    """
    def __init__(self, retriever: BaseRetriever, maxsize: int = 200) -> None:
        """Set up an empty cache around an existing retriever.

        Args:
            retriever: Retriever whose results are cached.
            maxsize: Maximum number of cached queries.
        """
        self.retriever = retriever
        self._cache: dict[tuple[str, int], list[ScoredChunk]] = (
            {}
        )  # (query and k)  O(1)
        self._maxsize = maxsize  # max of Questions
        self._order: list[tuple[str, int]] = (
            []
        )  # list of Questions sorted idnex of old is 0

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        """Return cached results if this (query, k) was seen,
        else search and cache.

        Args:
            query: The question to search for.
            k: Number of results to return.

        Returns:
            Ranked scored chunks.
        """
        cache_key = (query, k)
        if cache_key in self._cache:
            return self._cache[cache_key]  # return it from Ram very fast.

        results = self.retriever.search(query, k)
        self._add_to_cache(cache_key, results)
        return results

    def _add_to_cache(
        self, key: tuple[str, int], value: list[ScoredChunk]
    ) -> None:
        """Store a result, evicting the oldest entry when the cache is full.

        Args:
            key: The (query, k) pair.
            value: Results to store.
        """

        if len(self._order) >= self._maxsize:
            oldest_key = self._order.pop(0)
            del self._cache[oldest_key]

        self._cache[key] = value
        self._order.append(key)
