from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

openai_embedding = OpenAIEmbeddings()

model_name = 'BAAI/bge-small-zh-v1.5' #a dense embedding model that is suitable for Chinese
model_kwargs = {'device': 'cpu'} #you can also change it to gpu, if you use Nvidia
encode_kwargs = {'normalized': True}
bge_embedding = HuggingFaceEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)