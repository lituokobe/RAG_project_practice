from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from RAG_project_practice.llm_models.all_llm import llm
from RAG_project_practice.utils.log_utils import log


def transform_query(state):
    """
    Optimize user question, to generate more relevant query message.

    Args:
        state (dict): current graph state, including user question and retrieved results.

    Returns:
        state (dict): updated state, question is replaced with updated query message
    """
    log.info("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    transform_count = state.get("transform_count", 0)


    # prompt template
    system = """作为问题重写器，您需要将输入问题转换为更适合向量数据库检索的优化版本。\n
         请分析输入问题并理解其背后的语义意图/真实含义。"""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "这是初始问题: \n\n {question} \n 请生成一个优化后的问题。",
            ),
        ]
    )

    # create the chain the rewrite the question
    question_rewriter = (
            re_write_prompt
            | llm
            | StrOutputParser()
    )

    # rewrite the question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question, "transform_count": transform_count+1}