from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(  # openai的
    temperature=0,
    model='gpt-4o-mini')


web_search_tool = TavilySearchResults(max_results=2)