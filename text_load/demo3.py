import json
from langchain_core.documents import Document

#define a function that recovers content from json
def load_doc_from_json(json_file):
    """
    loads document from json file
    :param json_file:
    :return:
    """
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return Document(page_content=data['page_content'], metadata=data['metadata'])

if __name__ == '__main__':
    doc = load_doc_from_json('../datas/output/1_23.json')
    print(doc)
