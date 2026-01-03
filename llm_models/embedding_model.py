import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from RAG_project_practice.path import ENV_PATH

load_dotenv(ENV_PATH)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_embedding = OpenAIEmbeddings()

model_name = 'BAAI/bge-small-zh-v1.5' #a dense embedding model that is suitable for Chinese
model_kwargs = {'device': 'cpu'} #you can also change it to gpu, if you use Nvidia
encode_kwargs = {'normalized': True}
bge_embedding = HuggingFaceEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)