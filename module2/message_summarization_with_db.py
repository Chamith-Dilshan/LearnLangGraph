import os
from typing import Literal

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.postgres import PostgresSaver

""" 
Sync And Async

Synchronous (sync) execution runs code sequentially where each operation blocks execution until it finishes, 
while asynchronous (async) execution allows tasks to run concurrently without blocking.

In LangGraph and LangChain apps, the difference affects how nodes, checkpointers, and model calls are defined 
and invoked:

Sync (def / .invoke): Code executes line by line. If a node makes an LLM call, the entire program waits idly 
for the network response before moving on. Async (async def / await / .ainvoke): Code yields control 
during I/O wait times (like waiting for LLM responses or database reads/writes), allowing other concurrent 
tasks or requests to run simultaneously.

import asyncio
from langgraph.graph import MessagesState

# Async node definition using async/await
async def call_model(state: MessagesState):
    response = await model.ainvoke(state["messages"])
    return {"messages": [response]}
    
"""

load_dotenv()

class State(MessagesState):
    summary: str

llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model="openai/gpt-oss-20b",
        temperature=0,
    )

def call_model(state: State):
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]

    response = llm.invoke(messages)
    return {"messages": response}

def summarize_conversation(state: State):
    summary = state.get("summary", "")
    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)

    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

def should_continue(state: State) -> Literal["summarize_conversation", END]:
    messages = state["messages"]
    if len(messages) > 6:
        return "summarize_conversation"
    return END

workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node("summarize_conversation", summarize_conversation)

workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

DB_URI = os.getenv("DATABASE_URL")

if __name__ == "__main__":
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()
        graph = workflow.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "1"}}
        response = graph.invoke({"messages": [HumanMessage(content="Hello!")]}, config)
        print(response)