from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from RAG_project_practice.llm_models.all_llm import llm


class GradeHallucinations(BaseModel):
    """
    Grade the generated answers to see if there is any hallucination or not.
    """
    binary_score:str = Field(description="回答是否基于事实，取值为'yes'或'no'")


#Build the llm
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

#prompt template
system = """您是一个评估生成内容是否基于检索事实的评分器。\n 
    给出'yes'或'no'的二元评分。'yes'表示回答是基于\支持给定事实集的。"""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "事实集: \n\n{documents} \n\n 生成内容: {generation}")
    ]
)

#build the chain of checking hallucination
hallucination_grader_chain = hallucination_prompt | structured_llm_grader