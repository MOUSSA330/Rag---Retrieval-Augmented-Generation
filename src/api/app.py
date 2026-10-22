from typing import Any

from fastapi import FastAPI, HTTPException

from src.caching.cache import CachedRetriever, get_cached_lexical_retriever
from src.generation.generator import AnswerGenerator
from src.generator_service import GeneratorService
from src.retriever_service import RetrieverService
from src.validation import validate_k, validate_query

PROCESSED_DIR = "data/processed"
state: dict[str, Any] = {}

app = FastAPI(title="RAG against the machine")


@app.on_event("startup")
def startup_event() -> None:
    """Load the retriever and the generator once when the server starts.

    If the index is missing, the error is printed and the endpoints answer 503.
    """
    try:
        lexical = get_cached_lexical_retriever(PROCESSED_DIR)
        state["retriever_service"] = RetrieverService(CachedRetriever(lexical))
        state["generator_service"] = GeneratorService(AnswerGenerator())
    except FileNotFoundError as error:
        print(f"Error: {error}")


@app.on_event("shutdown")
def shutdown_event() -> None:
    """
    Release the loaded services when the server stops.
    """
    state.clear()


def _get_services() -> tuple[RetrieverService, GeneratorService]:
    """Return the loaded retriever and generator services.

    Returns:
        The retriever service and the generator service.

    Raises:
        HTTPException: 503 if the index was not loaded at startup.
    """
    retriever_service = state.get("retriever_service")
    generator_service = state.get("generator_service")
    if retriever_service is None or generator_service is None:
        raise HTTPException(
            status_code=503,
            detail="Index not loaded. Run the `index` command first.",
        )
    return retriever_service, generator_service


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the API is running."""
    return {"status": "ok"}


@app.get("/search")
def search(query: str, k: int = 5) -> dict[str, Any]:
    """Return the top-k sources for one question.

    Args:
        query: The question to search for.
        k: Number of sources to return.

    Raises:
        HTTPException: 400 if the query is empty or k is negative.
    """
    try:
        query = validate_query(query)
        k = validate_k(k)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    retriever_service, _ = _get_services()
    result = retriever_service.search(question=query, k=k)
    return (
        result.model_dump()
    )  # change to Json after is was a object. APi can anderstandi it


@app.get("/answer")
def answer(query: str, k: int = 5) -> dict[str, Any]:
    """Return a generated answer and its sources for one question.

    Args:
        query: The question to answer.
        k: Number of sources used as context.

    Raises:
        HTTPException: 400 if the query is empty or k is negative.
    """
    try:
        query = validate_query(query)
        k = validate_k(k)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    # MDN HTTP status codes 100 - 200 - 400...

    retriever_service, generator_service = _get_services()
    search_result = retriever_service.search(question=query, k=k)
    result = generator_service.generate_answer(search_result)
    return result.model_dump()
