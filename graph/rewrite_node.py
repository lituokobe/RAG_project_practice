from langchain_core.messages import HumanMessage

from RAG_project_practice.graph.get_human_message import get_last_human_message
from RAG_project_practice.graph.graph_state1 import AgentState
from RAG_project_practice.llm_models.all_llm import llm
from RAG_project_practice.utils.log_utils import log


def rewrite(state) :
    log.info("-------Converting Query-------")
    messages = state["messages"]
    question = get_last_human_message(messages).content

    msg = [
        HumanMessage(
            content = f""" \n 
            分析输入并尝试理解潜在的语义意图或含义。\n
            这是初始问题：
            \n ---------- \n
            {question}
            \n ---------- \n
            请提出一个改进后的问题：""",
        )
    ]

    response = llm.invoke(msg)
    return {"messages": [response]}
