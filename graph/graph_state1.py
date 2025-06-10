from typing import TypedDict, Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import Field, BaseModel


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages] #Sequence is a broader concept, List a child of Sequence


#data model class
class Grade(BaseModel):
    binary_score: str = Field(description = "相关性评分'yes'或者'no'")