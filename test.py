#define a workflow
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from RAG_PROJECT.graph2.generate_node2 import generate
from RAG_project_practice.graph.draw_png import draw_graph
from RAG_project_practice.graph2.grade_documents_node import grade_documents
from RAG_project_practice.graph2.graph_state2 import GraphState
from RAG_project_practice.graph2.retriever_node import retrieve
from RAG_project_practice.graph2.transform_query_node import transform_query
from RAG_project_practice.graph2.web_search_node import web_search

workflow = StateGraph(GraphState)

#add nodes
workflow.add_node('web_search', web_search)
workflow.add_node('retrieve', retrieve)
workflow.add_node('grade_documents', grade_documents)
workflow.add_node('generate', generate)
workflow.add_node('transform_query', transform_query)

workflow.add_edge(START, 'web_search')
workflow.add_edge('web_search', END)

graph = workflow.compile()
draw_graph (graph, 'graph_rag1-5.png')