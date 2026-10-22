from pathlib import Path

from src.models import MinimalSource

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant answering questions about a codebase. "
    "Use ONLY the provided context to answer. If the context does not "
    "contain enough information, say so honestly instead of guessing."
)


def fetch_source_text(source: MinimalSource) -> str:
    """Read the text of a source from its file.

    Args:
        source: Source to read (file path and character range).

    Returns:
        The text of the character range, or an error placeholder if the file
        cannot be read, so that generation can continue with the other sources.
    """
    try:
        content = Path(source.file_path).read_text(encoding="utf-8")
        return content[
            source.first_character_index:source.last_character_index
        ]
    except (FileNotFoundError, OSError, UnicodeDecodeError) as error:
        return f"[Error reading source: {error}]"


def build_context(sources: list[MinimalSource], max_sources: int = 3) -> str:
    """Assemble the text of the top retrieved sources into a single context.

    Args:
        sources: Retrieved sources, ranked by relevance.
        max_sources: Number of top sources kept in the context.

    Returns:
        The sources' text, each block labelled with its file path.
    """
    top_sources = sources[:max_sources]

    blocks = []
    for source in top_sources:
        text = fetch_source_text(source)
        blocks.append(f"### Source: {source.file_path}\n{text}")

    return "\n\n".join(blocks)


def build_prompt(question: str, context: str) -> list[dict[str, str]]:
    """Build the chat messages asking the model to answer
    from the context only.

    Args:
        question: The question to answer.
        context: Text assembled from the retrieved sources.

    Returns:
        System and user messages, ready for apply_chat_template.
    """
    user_content = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer based only on the context above."
    )
    # Chat Template because this model is trained like chatbot
    # so we gev it rules
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_content},
    ]
