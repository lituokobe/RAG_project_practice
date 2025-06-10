from RAG_project_practice.graph.graph_state1 import AgentState
from RAG_project_practice.llm_models.all_llm import llm
from RAG_project_practice.tools.retriever_tools import retriever_tool
from RAG_project_practice.utils.log_utils import log


def agent_node(state):
    log.info("-------Calling Agent-------")
    messages = state["messages"]

    model = llm.bind_tools([retriever_tool])
    response = model.invoke(messages)

    return {"messages": [response]}