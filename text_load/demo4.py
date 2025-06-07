from langchain_community.document_loaders import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader(
    file_path='../datas/md/operational_faq.md',
    mode='elements', #mode of loading, 'single': by default, load the file as a single file without structure
    #'elements': load the file with structure
)

docs = loader.load()
print(f'There are {len(docs)} documents.')

for i in range(10):

    print(docs[i].metadata)
    print(docs[i].page_content)

    print('--' * 50)