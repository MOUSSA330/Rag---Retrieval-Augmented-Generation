def validate_query(query: str) -> str:
    """Check that a query is not empty and strip surrounding whitespace.

    Args:
        query: Question typed by the user.

    Returns:
        The stripped query.

    Raises:
        ValueError: If the query is empty or only whitespace.
    """
    stripped = query.strip()
    if not stripped:
        raise ValueError("Query cannot be empty.")
    return stripped


def validate_k(k: int) -> int:
    """Check that k is not negative.

    Args:
        k: Number of results requested.

    Returns:
        The same k. A value of 0 is accepted and yields no results.

    Raises:
        ValueError: If k is negative.
    """
    if k <= 0:
        raise ValueError(f"k must be non-negative, got {k}.")
    return k
