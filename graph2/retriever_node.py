from RAG_project_practice.tools.retriever_tools import retriever
from RAG_project_practice.utils.log_utils import log


def retrieve(state):
    """
    retrieve relevant documents
    Args:
        state (dict): current graph state, including user question

    Returns:
        state (dict): updated state, added with documents including retrieved results
    """
    log.info("---去知识库中检索文档---")
    question = state["question"]
    # 文档检索
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
