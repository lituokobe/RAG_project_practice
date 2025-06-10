from langchain_core.documents import Document
from RAG_project_practice.llm_models.all_llm import web_search_tool
from RAG_project_practice.utils.log_utils import log


def web_search(state):
    """
    web search the question after optimization

    Args:
        state (dict): current graph state, including questions after optimize

    Returns:
        state (dict): state after update, documents are replaced with websearch content
    """
    log.info("---WEB SEARCH---")
    question = state["question"]  # get optimized question

    # Perform web search
    docs = web_search_tool.invoke({"query": question})  # call web search tool
    web_results = "\n".join([d["content"] for d in docs])  # join search result
    web_results = Document(page_content=web_results)  # convert format of documents

    return {"documents": web_results, "question": question} 