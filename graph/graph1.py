import uuid
from typing import Literal

from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from RAG_project_practice.graph.get_human_message import get_last_human_message
from RAG_project_practice.graph.agent_node import agent_node
from RAG_project_practice.graph.draw_png import draw_graph
from RAG_project_practice.graph.generate_node import generate
from RAG_project_practice.graph.graph_state1 import AgentState, Grade
from RAG_project_practice.graph.rewrite_node import rewrite
from RAG_project_practice.llm_models.all_llm import llm
from RAG_project_practice.tools.retriever_tools import retriever_tool
from RAG_project_practice.utils.log_utils import log
from RAG_project_practice.utils.print_utils import _print_event


#node of function to grade result
def grade_document(state) -> Literal["generate", "rewrite"]:
    """
    Check if the retrieved docs are relevant to the question.
    :param state: current state
    :return: str: result of judgement, if the docs are relevant
    """
    log.info("-------Checking Relevancy of Retrieved Documents-------")
    #LLM with structured output in a predesigned class type!!!
    llm_with_structured = llm.with_structured_output(Grade)

    #prompt template
    prompt = PromptTemplate(
        template="""你是一个评估检索文档与用户问题相关性的评分器。\n
                这是检索到的文档：\n\n {context} \n\n
                这是用户的问题：{question} \n
                如果文档包含与用户问题相关的关键词或语义含义，则评为相关。\n
                给出二元评分 'yes' 或 'no' 来表示文档是否与问题相关。""",
        input_variables=["context", "question"],
    )

    chain = prompt | llm_with_structured

    messages = state["messages"]
    last_message = messages[-1]

    question = get_last_human_message(messages).content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---Output: Relevant---")
        return "generate"

    else:
        print("---Output: Not Relevant---")
        print(score)
        return "rewrite"



#define a workflow
workflow = StateGraph(AgentState)

#add nodes
workflow.add_node('agent', agent_node)
workflow.add_node('retrieve', ToolNode([retriever_tool]))
workflow.add_node('rewrite', rewrite)
workflow.add_node('generate', generate)

#add edges
workflow.add_edge(START, 'agent')
workflow.add_conditional_edges(
    'agent',
    tools_condition,
    {
        'tools': 'retrieve',
        END: END
    }
)
workflow.add_conditional_edges(
    'retrieve',
    grade_document, # no need to add dictionary map and the results of grade_documents will be exactly the node names
)

workflow.add_edge('rewrite', 'agent')
workflow.add_edge('generate', END)


memory = MemorySaver()
graph = workflow.compile(checkpointer = memory)

#draw the graph
# draw_graph (graph, 'graph_rag1-2.png')

config = {
    "configurable": {
        "thread_id": str(uuid.uuid4()),
    }
}

_printed = set()

# execute workflow
while True:
    question = input('用户：')
    if question.lower() in ['q', 'exit', 'quit']:
        log.info('对话结束，拜拜！')
        break
    else:
        inputs = {
            "messages": [
                ("user", question),
            ]
        }
        events = graph.stream(inputs, config=config, stream_mode='values')
        #print the information
        for event in events:
            _print_event(event, _printed)

        #TODO: Try ask: EUV光刻机是什么？ 光刻胶有什么作用？
