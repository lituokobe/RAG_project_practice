import json

from IPython.core.display import HTML
from IPython.core.display_functions import display
from langchain_unstructured import UnstructuredLoader

pdf_file = "../datas/layout-parser-paper.pdf"
loader = UnstructuredLoader(
    file_path=pdf_file, #This can also be a list of urls
    strategy= 'hi_res', #'fast':cheap but inaccurate; 'hi_res': slow and expensive
    partition_via_api=False, #True: paid; False: free but need local parser
    #coordinates=True, #need to keep original coordinate locations of the texts, or not
    #if run locally, we need to remove 'coordinates', as it will raise an error
    #api_key=,#only company users can register account and have API keys now
)


docs = []
counter = 0

#define a function to output the json
def write_json(data, file_name):
    with open('../datas/output/'+file_name, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

#append all the docs parse to the list of docs, and invoke the function to output the json
for doc in loader.lazy_load():
    docs.append(doc)
    json_file_name = str(doc.metadata.get('page_number'))+'_'+str(counter)+'.json' #output as json and define the nmae
    counter+=1
    write_json(doc.model_dump(), json_file_name) #model_dump converts the doc to a json

#check the docs list
# print(docs)
print(f'There are {str(len(docs))} documents parsed.') #this is the page number of the PDF document
print(docs[0].metadata)
print(docs[0].page_content)# check the first page
print('------------------')

#There is a table and code section in page 5, let's take a look at how it is parsed
segments = [
    doc.metadata
    for doc in docs
    if doc.metadata.get("page_number") == 5 and doc.metadata.get("category") == "Table"
]
print(f'The table is:')
print(segments)
# display(HTML(segments[0]["text_as_html"]))

#display(HTML(segments[0].get("text_as_html", segments[0].get("text", "No HTML available"))))
