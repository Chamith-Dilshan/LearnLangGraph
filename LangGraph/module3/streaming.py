import asyncio
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph_sdk import get_client

load_dotenv()

# llm
llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="openai/gpt-oss-20b",
    temperature=0,
)


# State
class AgentState(MessagesState):
    summary: str


# Define the logic to call the model
def call_model(state: AgentState):
    # Get summary if it exists
    summary = state.get("summary", "")

    # If there is summary, then we add it
    if summary:
        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"

        # Append summary to any newer messages
        messages = [SystemMessage(content=system_message)] + state["messages"]

    else:
        messages = state["messages"]

    response = llm.invoke(messages)
    return {"messages": response}


def summarize_conversation(state: AgentState):
    # First, we get any existing summary
    summary = state.get("summary", "")

    # Create our summarization prompt
    if summary:
        # A summary already exists
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )

    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)

    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}


# Determine whether to end or summarize the conversation
def should_continue(state: AgentState) -> Literal["summarize_conversation", END]:
    """Return the next node to execute."""

    messages = state["messages"]

    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"

    # Otherwise we can just end
    return END


# Define a new graph
workflow = StateGraph(AgentState)
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

# Set the entrypoint as conversation
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

# Compile
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)


async def astream_events_graph1():
    config = {"configurable": {"thread_id": "3"}}
    input_message = HumanMessage(content="Tell me about the latest FIFA teams")
    async for event in graph.astream_events(
        {"messages": [input_message]}, config, version="v2"
    ):
        print(
            f"Node: {event['metadata'].get('langgraph_node', '')}. Type: {event['event']}. Name: {event['name']}"
        )


async def astream_events_graph2():
    node_to_stream = "conversation"
    config = {"configurable": {"thread_id": "4"}}
    async for event in graph.astream_events(
        {"messages": [input_message]}, config, version="v2"
    ):
        # Get chat model tokens from a particular node
        if (
            event["event"] == "on_chat_model_stream"
            and event["metadata"].get("langgraph_node", "") == node_to_stream
        ):
            print(event["data"])


async def astream_events_graph3():
    config = {"configurable": {"thread_id": "5"}}
    node_to_stream = "conversation"
    async for event in graph.astream_events(
        {"messages": [input_message]}, config, version="v2"
    ):
        # Get chat model tokens from a particular node
        if (
            event["event"] == "on_chat_model_stream"
            and event["metadata"].get("langgraph_node", "") == node_to_stream
        ):
            data = event["data"]
            print(data["chunk"].content, end="|")


async def use_langgraph_api():
    # This is the URL of the local development server
    # first you need to start the server by running `langgraph dev`
    URL = "http://127.0.0.1:2024"
    client = get_client(url=URL)

    # Search all hosted graphs
    assistants = await client.assistants.search()

    # Create a new thread
    thread = await client.threads.create()

    input_message = HumanMessage(content="Multiply 2 and 3")

    async for event in client.runs.stream(
        thread["thread_id"],
        assistant_id="streaming",
        input={"messages": [input_message]},
        stream_mode="messages",
    ):
        # Handle metadata events
        if event.event == "metadata":
            print(f"Metadata: Run ID - {event.data['run_id']}")
            print("-" * 50)

        # Handle partial message events
        elif event.event == "messages/partial":
            for data_item in event.data:
                # Process user messages
                if "role" in data_item and data_item["role"] == "user":
                    print(f"Human: {data_item['content']}")
                else:
                    # Extract relevant data from the event
                    tool_calls = data_item.get("tool_calls", [])
                    invalid_tool_calls = data_item.get("invalid_tool_calls", [])
                    content = data_item.get("content", "")
                    response_metadata = data_item.get("response_metadata", {})

                    if content:
                        print(f"AI: {content}")

                    if tool_calls:
                        print("Tool Calls:")
                        print(format_tool_calls(tool_calls))

                    if invalid_tool_calls:
                        print("Invalid Tool Calls:")
                        print(format_tool_calls(invalid_tool_calls))

                    if response_metadata and response_metadata.get("finish_reason"):
                        print(
                            f"Response Metadata: Finish Reason - {response_metadata['finish_reason']}"
                        )
            print("-" * 50)


def format_tool_calls(tool_calls):
    """
    Format a list of tool calls into a readable string.

    Args:
        tool_calls (list): A list of dictionaries, each representing a tool call.
            Each dictionary should have 'id', 'name', and 'args' keys.

    Returns:
        str: A formatted string of tool calls, or "No tool calls" if the list is empty.

    """

    if tool_calls:
        formatted_calls = []
        for call in tool_calls:
            formatted_calls.append(
                f"Tool Call ID: {call['id']}, Function: {call['name']}, Arguments: {call['args']}"
            )
        return "\n".join(formatted_calls)
    return "No tool calls"


async def langgraph_api_streaming():
    # LangGraph SDK client streaming

    # This is the URL of the local development server
    # first you need to start the server by running `langgraph dev`
    URL = "http://127.0.0.1:2024"
    client = get_client(url=URL)

    # Create a new thread
    thread = await client.threads.create()

    input_message = HumanMessage(content="Multiply 2 and 3")

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id="streaming",
        input={"messages": [input_message]},
        stream_mode="messages-tuple",
    ):
        if chunk.event != "messages":
            continue

        message_chunk, metadata = chunk.data
        if message_chunk.get("content"):
            print(message_chunk["content"], end="", flush=True)


if __name__ == "__main__":
    """ Streaming full state of the graph 
        
    .stream and .astream are sync and async methods for streaming back results.

    LangGraph supports a few different streaming modes for graph state.

    values: This streams the full state of the graph after each node is called.
    updates: This streams updates to the state of the graph after each node is called.
        
    """

    # Create a thread
    config = {"configurable": {"thread_id": "1"}}

    # Start conversation
    for chunk in graph.stream(
        {"messages": [HumanMessage(content="hi! I'm Lance")]},
        config,
        stream_mode="updates",
    ):
        print(chunk)
        chunk["conversation"]["messages"].pretty_print()

    # Start conversation, again
    config2 = {"configurable": {"thread_id": "2"}}
    input_message = HumanMessage(content="hi! I'm Lance")
    for event in graph.stream(
        {"messages": [input_message]}, config2, stream_mode="values"
    ):
        for m in event["messages"]:
            m.pretty_print()
        print("---" * 25)

    # Direct graph invocation with token streaming
    for chunk in graph.stream(
        {"messages": [HumanMessage(content="Tell me a joke")]},
        stream_mode="messages",
        version="v2",
    ):
        if chunk["type"] == "messages":
            message_chunk, metadata = chunk["data"]
            if message_chunk.content:
                print(message_chunk.content, end="", flush=True)

    print("@@@@@" * 50)

    """ Streaming tokens 
        
        We often want to stream more than graph state.
        We can do this using the .astream_events method, which streams back events as they happen inside nodes!

        Each event is a dict with a few keys:

        event: This is the type of event that is being emitted.
        name: This is the name of event.
        data: This is the data associated with the event.
        metadata: Containslanggraph_node, the node emitting the event.
        
    """

    # asyncio.run(astream_events_graph1())
    #
    # asyncio.run(astream_events_graph2())
    #
    # asyncio.run(astream_events_graph3())

    """ Using LangGraph API and LangGraph_sdk to steam output"""

    # asyncio.run(use_langgraph_api())
