In this RAG practice, we use Milvus as the vector database as it is the most popular one now.
- In demo, we connect to local Milvus Lite and perform some basic database operation.
- In demo1, we connect to a Milvus Standalone on a server (local machine) and perform some basic database operation.


Under text-load:
- In demo1, we used basic PyPDFLoader to parse a PDF document.
- In demo2, we used langchain_unstructured (local) to parse a PDF documents and output the jsons.
- In demo3, we defined a function to recover the content from a generated json file.
- In demo4, we used langchain_unstructured to load a markdown file.

Under documents:
- In markdown_parser, we defined a chunking and loading class for markdown documents, it can parse and load the document, merge paragraphs under one title, and chunk long content.
- In milvus_db, we created a class with functions to create a collection with index of dense and sparse vectors.
- In write_milvus, 

Under test_vector:
- In demo1, we practiced dense embedding with bge.
- In demo2, we practiced sparse embedding with BM25 using index.