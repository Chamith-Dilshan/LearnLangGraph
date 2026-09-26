import os

from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq


def chat_llm():
    llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model="openai/gpt-oss-20b",
        temperature=0,
    )
    msg = HumanMessage(content="Hello, how are you?", name="John")

    messages = [msg]

    result = llm.invoke(messages)

    return result

def search_web():
    tavily_search = TavilySearch(max_results=5)

    data = tavily_search.invoke("What is LangGraph?")
    print("data: ",data)

    return data.get("results",data)

if __name__ == '__main__':
    messages = chat_llm()
    print("Message",messages)
    print("Metadata", messages.response_metadata)

    print("/" * 20)

    search_docs = search_web()
    print("Search Docs: ",search_docs)
