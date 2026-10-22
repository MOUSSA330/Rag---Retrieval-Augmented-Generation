import uuid
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):  # where we find the data
    """
    A location in the corpus: a file and a character range within it.
    """
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """
    A question without an answer; question_id defaults to a new UUID.
    """
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """
    A question with its ground-truth sources and reference answer.
    """
    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """
    A dataset of questions, answered or not.
    """
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """
    The sources retrieved for one question.
    """
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """
    Search results for one question, completed with a generated answer.
    """
    answer: str


class StudentSearchResults(BaseModel):
    """
    Output of search_dataset: results for every question, and the k used.
    """
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """
    Output of answer_dataset: answers for every question, and the k used.
    """
    search_results: list[MinimalAnswer]
    k: int
