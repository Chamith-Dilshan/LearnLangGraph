import asyncio
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt.tool_node import ToolNode
from langgraph_sdk import get_client

load_dotenv()


def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


# This will be a tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


def divide(a: int, b: int) -> float:
    """Divide a by b.

    Args:
        a: first int
        b: second int
    """
    return a / b


tools = [multiply, add, divide]
llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="openai/gpt-oss-20b",
    temperature=0,
)
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(
    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
)


# Agent
def agent(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


# Graph
builder = StateGraph(MessagesState)

# Define Nodes
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))

# Define edges
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")  # Enable ReAct nature of the Agent
builder.add_edge("agent", END)

memory = MemorySaver()
graph = builder.compile(interrupt_before=["agent"], checkpointer=memory)


async def use_langgraph_api():
    client = get_client(url="http://127.0.0.1:2024")

    initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}
    thread = await client.threads.create()

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id="editing_state_and_human_feedback",
        input=initial_input,
        stream_mode="values",
        interrupt_before=["agent"],
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        messages = chunk.data.get("messages", [])
        if messages:
            print(messages[-1])
        print("-" * 50)

    current_state = await client.threads.get_state(thread["thread_id"])

    last_message = current_state["values"]["messages"][-1]

    last_message["content"] = "No, actually multiply 3 and 3!"

    await client.threads.update_state(thread["thread_id"], {"messages": last_message})

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id="editing_state_and_human_feedback",
        input=None,
        stream_mode="values",
        interrupt_before=["agent"],
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        messages = chunk.data.get("messages", [])
        if messages:
            print(messages[-1])

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id="editing_state_and_human_feedback",
        input=None,
        stream_mode="values",
        interrupt_before=["agent"],
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        messages = chunk.data.get("messages", [])
        if messages:
            print(messages[-1])
        print("-" * 50)


if __name__ == "__main__":
    # Draw and save graph image
    os.makedirs("images", exist_ok=True)
    file_path = os.path.join("images", "human_feedback.png")

    if not os.path.exists(file_path):
        png_data = graph.get_graph().draw_mermaid_png()
        with open(file_path, "wb") as f:
            f.write(png_data)
        print(f"Image saved to {file_path}")
    else:
        print(f"Image already exists at {file_path}")

    # Input
    initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}

    # Thread
    thread = {"configurable": {"thread_id": "1"}}

    # Run the graph until the first interruption
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

    state = graph.get_state(thread).values
    print(f"This is before update: {state}")

    graph.update_state(
        thread, {"messages": [HumanMessage(content="No, actually multiply 3 and 3!")]}
    )

    new_state = graph.get_state(thread).values
    print(f"This is after update: {new_state}")
    for m in new_state["messages"]:
        m.pretty_print()

    # simply by passing None and allowing it to proceed from the current state.
    for event in graph.stream(None, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

    # Now, we're back at the agent, which has our breakpoint.
    # We can again pass None to proceed.
    for event in graph.stream(None, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

    # How to interrupt the graph and update the state with langgraph SDK
    # asyncio.run(use_langgraph_api())
