
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from RAG_project_practice.llm_models.all_llm import llm


#route for query: based on user's question to decide which search strategy (web search or RAG)
# data model
class RouteQuery(BaseModel):
    """route user's query to the most relevant data source"""
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="根据用户问题选择将其路由到向量知识库或网络搜索",
    )


structured_llm_router = llm.with_structured_output(RouteQuery)

# prompt template
system = """你是一个擅长将用户问题路由到向量知识库或网络搜索的专家。
向量知识库包含与半导体材料，芯片制造，光刻技术相关的文档。
对于这些主题的问题请使用向量知识库，其他情况使用网络搜索。"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),  # system prompt
        ("human", "{question}"),  # placeholder for user question
    ]
)

# create question route chain
question_router_chain = route_prompt | structured_llm_router


# # testing
# print(  # technical - RAG
#     question_router_chain.invoke(
#         {"question": "什么是EUV光刻技术?"}
#     )
# )
# print(  # non technical - web search
#     question_router_chain.invoke({"question": "今天，长沙的天气怎么样?"})
# )