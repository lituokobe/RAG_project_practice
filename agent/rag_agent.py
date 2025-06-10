from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.tools import create_retriever_tool
from langchain_community.chat_message_histories import ChatMessageHistory
from RAG_project_practice.documents.milvus_db import MilvusVectorSave
from RAG_project_practice.llm_models.all_llm import llm
from RAG_project_practice.tools.retriever_tools import retriever_tool

#prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a smart assistant, try to call tools to answer user's question."),
    MessagesPlaceholder(variable_name='chat_history', optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name='agent_scratchpad', optional=True),
    """
    agent_scratchpad is a placeholder used to store the intermediate steps of an agent's reasoning process.
    It keeps track of the agent's actions, tool invocations, and observations during execution.
    """
])



agent = create_tool_calling_agent(llm, [retriever_tool], prompt)

#executor is the last agent function in langchain by June 2025,
#all the features will be moved to langgraph very soon.
executor = AgentExecutor(agent=agent, tools=[retriever_tool])



#Directly use the agent
# res1 = executor.invoke({'input': '什么是EUV光刻机？'})
# print(res1)




#Let the agent record chat history
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

agent_with_history = RunnableWithMessageHistory(
    executor,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
)

rep2= agent_with_history.invoke(
    {'input': '什么是EUV光刻机？'},
    config={'configurable': {'session_id':'luis123'}},
)

print(rep2)