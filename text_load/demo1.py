from langchain_community.document_loaders import PyPDFLoader

pdf_file = "../datas/layout-parser-paper.pdf"
loader = PyPDFLoader(file_path=pdf_file) #page by page parsing, one page is one document object

docs = loader.load()

print(docs)
print(len(docs)) #this is the page number of the PDF document
print(docs[0].metadata, docs[0].page_content)