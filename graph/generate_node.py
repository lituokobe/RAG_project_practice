from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from RAG_project_practice.graph.get_human_message import get_last_human_message
from RAG_project_practice.llm_models.all_llm import llm
from RAG_project_practice.utils.log_utils import log


def generate(state):
    log.info("-------Generating Final Answer-------")
    messages = state["messages"]
    question = get_last_human_message(messages).content
    docs = messages[-1].content
    # last_message = messages[-1] #the last message is the result from RAG retrieval, it could be multiple documents
    #
    # docs = last_message.content

    prompt = PromptTemplate(
        template="你是一个问答任务助手。请根据以下检索到的上下文内容回答问题。如果不知道答案，请直接说明。回答保持简洁。\n问题：{question} \n上下文：{context} \n回答：",
        input_variables=["question", "context"],
    )


    #process chain
    rag_chain = prompt | llm |StrOutputParser()

    #execute
    response = rag_chain.invoke(
        {
            "context": docs,
            "question": question,
        }
    )

    ai_message = AIMessage(content=response)
    return {"messages": [ai_message]}
