from typing import List
from langchain_experimental.text_splitter import SemanticChunker

from RAG_project_practice.llm_models.embedding_model import openai_embedding
from RAG_project_practice.utils.log_utils import log
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document


class MarkdownParser():
    """
    for parsing and chunking markdown
    """
    #iniitiate a semantic chunker, using openai_embedding
    def __init__(self):
        self.text_plitter = SemanticChunker(
            openai_embedding,
            breakpoint_threshold_type="percentile"
        )


    def text_chunker(self, datas: List[Document]) -> List[Document]:
        new_docs = []
        for d in datas:
            if len(d.page_content)>500: # one document's content is more than 8000, chunk it again with sematic method
                new_docs.extend(self.text_plitter.split_documents([d]))
                """
                The extend() method is used to add elements from an iterable
                (like another list, tuple, or set) to the end of the current list.
                Unlike append(), which adds the entire iterable as a single element,
                extend() iterates over the provided iterable and adds each element individually.
                """
                # continue #The continue statement in your code is used to skip the rest of the loop's body and move to the next iteration immediately.
                ## If you don't have the else: below, you should have 'continue' here to avoid unnecessary new_docs.append(d)

            else:
                new_docs.append(d)
        return new_docs

    def parse_markdown_to_documents(self, md_file:str, encoding='utf-8') -> List[Document]:
        documents = self.parse_markdown(md_file)
        log.info(f'file length after parsing: {len(documents)}')

        merged_documents = self.merge_title_content(documents)
        log.info(f'file length after merging: {len(merged_documents)}')

        chunked_documents = self.text_chunker(merged_documents)
        log.info(f'file length after semantic chunking: {len(chunked_documents)}')

        return chunked_documents


    #define a function to chunk and load the markdown
    def parse_markdown(self, md_file: str)-> List[Document]:
        loader = UnstructuredMarkdownLoader(
            file_path=md_file,
            mode='elements',
            strategy='fast'
        )
        docs = []
        for doc in loader.lazy_load():
            docs.append(doc)
        return docs

    #Sometimes, a markdown is chunked with too many detailed pieces, e.g. each paragraph is chunked into one small document.
    #We need to merge these small documents if their original paragraphs are under one title for example.
    #Here we define a functino to do this.
    def merge_title_content(self, datas: List[Document])-> List[Document]:
        merged_data = []
        parent_dict = {} #dictionary to store all parent documents
        for document in datas:
            metadata = document.metadata
            # as language is parsed in list, which is not supported by Milvus database. We simply delete it.
            if 'languages' in metadata:
                metadata.pop('languages')

            parent_id = metadata.get('parent_id', None)
            category = metadata.get('category', None)
            element_id = metadata.get('element_id', None)

            if category == 'NarrativeText' and parent_id is None:
                merged_data.append(document)
            if category == 'Title':
                document.metadata['title'] = document.page_content #customize to add 'title' to metadata
                if parent_id in parent_dict: #meaning the title document is not top level title as it has a parent, add its parent document (a higher level title only to the content)
                    document.page_content = parent_dict[parent_id].page_content + ' -> ' + document.page_content
                parent_dict[element_id] = document
            if category != 'Title' and parent_id:
                parent_dict[parent_id].page_content = parent_dict[parent_id].page_content + ' ' + document.page_content
                # change the category to a newly created value 'content'.
                # It's easier for us to filter them out later, as these documents are usually more important.
                parent_dict[parent_id].metadata['category'] = 'content'

        if parent_dict is not None:
            merged_data.extend(parent_dict.values())

        return merged_data




if __name__ == '__main__':
    file_path = '../datas/md/operational_faq.md'
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(file_path)
    for item in docs:
        print(f"Metadata:{item.metadata}")
        print(f"Title:{item.metadata.get('title'), None}")
        print(f"Content:{item.page_content}\n")
        print("-------"*10)