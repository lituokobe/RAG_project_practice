from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from RAG_project_practice.llm_models.all_llm import llm


def generate(state):
    question = state["question"]
    documents = state['documents']

    prompt = PromptTemplate(
        template="你是一个问答任务助手。请根据以下检索到的上下文内容回答问题。如果不知道答案，请直接说明。回答保持简洁。\n问题：{question} \n上下文：{context} \n回答：",
        input_variables=["question", "context"],
    )

    # after-process function - format documents after retrieval
    def format_docs(docs):
        """Merge all documents into one string with 2 change lines as separator"""
        if isinstance(docs, list):
            return "\n\n".join(doc.page_content for doc in docs)
        else:
            return "\n\n" + docs.page_content


    #process chain
    rag_chain = prompt | llm | StrOutputParser()

    #execute
    generation = rag_chain.invoke(
        {
            "context": format_docs(documents),
            "question": question,
        }
    )


    return {
        "documents": documents,
        "question": question,
        "generation": generation
    }
