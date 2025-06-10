from typing import TypedDict, List

from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    illustrate the state information of graph process
    question: text of user question
    transform_count: numbers of transforming queries
    generation: generated text from llm
    documents: retrieved documents
    """
    question: str #store current user question
    transform_count: int #numbers of transforming queries
    generation: str #store llm's generated answers
    documents: List[Document] #store retrieved document list