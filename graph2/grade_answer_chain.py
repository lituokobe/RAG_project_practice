from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from RAG_project_practice.llm_models.all_llm import llm

# data model
class GradeAnswer(BaseModel):
    """Evaluate if the answer solve user's question's binary grading model"""

    binary_score: str = Field(
        description="回答是否解决了问题，取值为'yes'或'no'"
    )


# build the llm
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# prompt template
system = """您是一个评估回答是否解决用户问题的评分器。\n
     给出'yes'或'no'的二元评分。'yes'表示:回答确实解决了该问题。"""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "用户问题: \n\n {question} \n\n 生成回答: {generation}"),
    ]
)

# build the chain
answer_grader_chain = answer_prompt | structured_llm_grader
