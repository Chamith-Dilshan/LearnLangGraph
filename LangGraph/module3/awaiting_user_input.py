import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt.tool_node import ToolNode

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


# no-op node that should be interrupted on
def human_feedback(state: MessagesState):
    pass


# Graph
builder = StateGraph(MessagesState)

# Define Nodes
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))
builder.add_node("human_feedback", human_feedback)

# Define edges
builder.add_edge(START, "human_feedback")
builder.add_edge("human_feedback", "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "human_feedback")
builder.add_edge("agent", END)

memory = MemorySaver()
graph = builder.compile(interrupt_before=["human_feedback"], checkpointer=memory)


if __name__ == "__main__":
    # Draw and save graph image
    os.makedirs("images", exist_ok=True)
    file_path = os.path.join("images", "awaiting_input.png")

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

    # Get user input
    user_input = input("Tell me how you want to update the state: ")

    # We now update the state as if we are the human_feedback node
    graph.update_state(thread, {"messages": user_input}, as_node="human_feedback")

    # Continue the graph execution
    for event in graph.stream(None, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

    # Continue the graph execution
    for event in graph.stream(None, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()
