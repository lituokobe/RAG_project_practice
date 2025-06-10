from RAG_project_practice.utils.log_utils import log


def draw_graph(graph, file_name: str):
    try:
        mermaid_code = graph.get_graph().draw_mermaid_png()
        with open(file_name, 'wb') as f:
            f.write(mermaid_code)
    except Exception as e:
        log.exception(e)