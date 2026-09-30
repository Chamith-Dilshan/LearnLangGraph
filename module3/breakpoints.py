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


tools = [add, multiply, divide]
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
graph = builder.compile(interrupt_before=["tools"], checkpointer=memory)


async def use_langgraph_api():
    client = get_client(url="http://127.0.0.1:2024")

    initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}
    thread = await client.threads.create()

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id="agent",
        input=initial_input,
        stream_mode="values",
        interrupt_before=["tools"],
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        messages = chunk.data.get("messages", [])
        if messages:
            print(messages[-1])
        print("-" * 50)

    # Resume the task

    async for chunk in client.runs.stream(
        thread["thread_id"],
        "agent",
        input=None,
        stream_mode="values",
        interrupt_before=["tools"],
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        messages = chunk.data.get("messages", [])
        if messages:
            print(messages[-1])
        print("-" * 50)


if __name__ == "__main__":
    # Draw and save graph image
    os.makedirs("images", exist_ok=True)
    file_path = os.path.join("images", "breakpoint.png")

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

    state = graph.get_state(thread)
    print(state.next)

    # Get user feedback
    user_approval = input("Do you want to call the tool? (yes/no): ")

    # Check approval
    if user_approval.lower() == "yes":
        # If approved, continue the graph execution
        for event in graph.stream(None, thread, stream_mode="values"):
            event["messages"][-1].pretty_print()

    else:
        print("Operation cancelled by user.")

    """ When you work with langgraph api, you can also pass interrupt_before to the stream method directly.
        First locally run the langgraph api with "langgraph dev" 
        You may need to remove the in memory before running it from the api.
    """
    # asyncio.run(use_langgraph_api())
