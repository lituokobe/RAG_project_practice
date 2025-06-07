# Using a Distributed, Multi-Process Approach to Write Massive Data into the Milvus Database
import os
from multiprocessing import Queue
from RAG_project_practice.documents.markdown_parser import MarkdownParser
from RAG_project_practice.utils.log_utils import log


def file_parser_process(dir_path: str, output_queue: Queue, batch_size: int = 20):
    #using batching can decrease the usage of RAM
    """
    Process 1: parse all markdown files under the directory and put them into queue in batches
    :param dir_path:
    :param output_queue:
    :param batch_size:
    :return:
    """
    log.info(f"Parsing process starts and scan the directory: {dir_path}.")

    #get all .md file
    md_files = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.endswith(".md")
    ]

    if not md_files:
        log.warning("Warning: no markdown files found.")
        output_queue.put(None) #.put() is a function to insert content to Queue instance
        return

    parser = MarkdownParser()
    doc_batch = []
    for file_path in md_files:
        try:
            docs = parser.parse_markdown_to_documents(file_path)
            if docs:
                doc_batch.extend(docs)
            #put it to queue in batches
            if len(doc_batch) >= batch_size:
                output_queue.put(doc_batch.copy()) #insert a copy to the queue
                doc_batch.clear() #clear all the batches in current buffer
        except Exception as e:
            log.warning(f"Error while parsing {file_path}: {str(e)}")
            log.exception(e)

    #insert leftover docs
    if doc_batch:
        output_queue.put(doc_batch)
        #no need to use .copy() and .clear() because we won't use doc_batch anymore

    #finishing
    output_queue.put(None)
    log.info(f"Parsing process ends and processed {len(md_files)} documents.")


