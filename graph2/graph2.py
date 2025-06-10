import uuid
from pprint import pprint

from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from RAG_project_practice.graph.draw_png import draw_graph
from RAG_project_practice.graph2.generate_node2 import generate
from RAG_project_practice.graph2.grade_answer_chain import answer_grader_chain
from RAG_project_practice.graph2.grade_documents_node import grade_documents
from RAG_project_practice.graph2.grade_hallucinations_chain import hallucination_grader_chain
from RAG_project_practice.graph2.graph_state2 import GraphState
from RAG_project_practice.graph2.query_route_chain import question_router_chain
from RAG_project_practice.graph2.retriever_node import retrieve
from RAG_project_practice.graph2.transform_query_node import transform_query
from RAG_project_practice.graph2.web_search_node import web_search
from RAG_project_practice.utils.log_utils import log

#route function right after START
def route_question(state):
    """
    Route the question to web search or RAG
    Args:
        state (dict): current graph state, including user question
    Returns:
        str: name of next node, web_search or vectorstore
    """
    log.info("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router_chain.invoke({"question": question})

    # decide next node based on router result
    if source.datasource == "web_search":
        log.info("---Route to web search---")
        return "web_search"
    elif source.datasource == "vectorstore":
        log.info("---Route to RAG---")
        return "vectorstore"

def decide_to_generate(state):
    """
    Decide to generated answer or to reoptimize question
    :param state: current graph state, including filtered documents
    :return: name of next node (transform_query or generate)
    """
    log.info("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]
    transform_count = state.get("transform_count", 0)

    if not filtered_documents:
        if transform_count >=2:
            log.info("-----Decision: all documents are not relevant, and looped twice, need to redo web search---")
            return "web_search"
        log.info("-----Decision: all documents are not relevant, need to convert questions---")
        return "transform_query"
    else:
        log.info("-----Decision: generate final answer---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Evaluate if the generated result is based on documents and solve the problem from the question.
    :param state: current graph state, including user question, documents, and generated results
    :return: name of next node (useful, not useful or not supported)
    """
    log.info("---Check if generated result has hallucination---")
    question = state["question"]
    documents = state["documents"]
    generation =state["generation"]

    #check if generated result is based on documents
    score = hallucination_grader_chain.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score

    if grade == 'yes':
        log.info("---Decision: generated result is based on documents---")

        #check if generated result solves the problem from the question
        log.info("---Check if generated result solves the problem from the question---")
        score = answer_grader_chain.invoke({"question": question, "generation": generation})
        grade = score.binary_score

        if grade == 'yes':
            log.info("---Decision: generated result solves the problem from the question---")
            return "useful"

        else:
            log.info("---Decision: generated result does not solve the problem from the qeustion---")
            return "not useful"
    else:
        log.info("---Decision: generated result is not based on documents, will try again---")
        return "not supported"



#define a workflow
workflow = StateGraph(GraphState)

#add nodes
workflow.add_node('web_search', web_search)
workflow.add_node('retrieve', retrieve)
workflow.add_node('grade_documents', grade_documents)
workflow.add_node('generate', generate)
workflow.add_node('transform_query', transfor   m_query)

#add edges
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search":"web_search",
        "vectorstore":"retrieve",
    }
)

workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    'grade_documents',
    decide_to_generate
)

workflow.add_conditional_edges(
    'generate',
    grade_generation_v_documents_and_question,
    {
        "not supported":"generate",
        "useful":END,
        "not useful":"transform_query",
    }
)

workflow.add_edge("transform_query", "retrieve")

#compilation
# memory = MemorySaver()
# graph = workflow.compile(checkpointer = memory)
graph = workflow.compile()

#draw the graph
# draw_graph (graph, 'graph_rag1-4.png')

_printed = set()

# execute workflow
while True:
    question = input('User:')
    if question.lower() in ['q', 'exit', 'quit']:
        log.info('The chat is over, bye-bye!')
        break
    else:
        inputs = {
            "question": question
        }
        for output in graph.stream(inputs):
            for key, value in output.items():
                # 打印当前节点名称
                pprint(f"Node '{key}':")  # 显示当前执行的节点名称
                # 可选：打印每个节点的完整状态信息
                # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
            pprint("\n---\n")  # 节点分隔线

        # 打印最终生成结果
        pprint(value["generation"])  # 输出最终生成的回答内容

        #TODO: Try ask: EUV光刻机是什么？ 光刻胶有什么作用？ 清洗晶圆的技术有哪些？
        #TODO：Try ask: 谁发明了EUV光刻机？DUV光刻机有哪些缺点？



