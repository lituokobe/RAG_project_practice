from RAG_project_practice.graph2.grader_chain import retrieval_grader_chain
from RAG_project_practice.utils.log_utils import log


def grade_documents(state):
    """
    evaluate the relevancy between retrieved documents and the question

    Args:
        state (dict): current graph state, including question and retrieved results

    Returns:
        state (dict): updated state, only relevant documents will be kept
    """
    log.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]  # get user question
    documents = state["documents"]  # get documents to evaluate

    # score the documents and filter them
    filtered_docs = []  # initiate filtered document list
    for d in documents:
        score = retrieval_grader_chain.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            log.info("---GRADE: Retrieved Document is Relevant--")
            filtered_docs.append(d)  # add it to the filtered document list
        else:
            log.info("---GRADE: Retrieved Document is NOT Relevant, Dump it---")
            continue
    return {"documents": filtered_docs, "question": question}  # Only return the state with filtered documents