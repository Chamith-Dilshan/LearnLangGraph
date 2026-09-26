import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# Agent state
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage],add_messages]


# Tools
def multiply(a: int, b:int):
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def add(a:int, b:int):
    """Add a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def subtract(a:int, b:int):
    """Subtract b from a.

    Args:
        a: first int
        b: second int
    """
    return a - b

def divide(a:int, b:int):
    """Divide a by b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [multiply, add, subtract, divide]
llm =  ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model="openai/gpt-oss-20b",
        temperature=0,
    )

llm_with_tools = llm.bind_tools(tools,parallel_tool_calls=False)

# System message
sys_msg = SystemMessage(
    content=(
        "You are a helpful assistant tasked with performing arithmetic on a set of inputs. "
        "You MUST use the provided tools for EVERY calculation step. "
        "Never calculate values mentally or provide final answers without executing the required tool calls."
    )
)

# Agent Node
def agent(state: AgentState):
    return {"messages": [llm_with_tools.invoke([sys_msg]+ state["messages"])]}

# Graph
builder = StateGraph(AgentState)

# Define Nodes
builder.add_node("agent",agent)
builder.add_node("tools",ToolNode(tools))

# Define edges
builder.add_edge(START,"agent")
builder.add_conditional_edges("agent",tools_condition)
builder.add_edge("tools","agent") # Enable ReAct nature of the Agent
builder.add_edge("agent",END)

react_graph = builder.compile()

if __name__ == "__main__":
    # Draw and save graph image
    os.makedirs("images", exist_ok=True)
    file_path = os.path.join("images", "react_agent_graph.png")

    if not os.path.exists(file_path):
        png_data = react_graph.get_graph().draw_mermaid_png()
        with open(file_path, "wb") as f:
            f.write(png_data)
        print(f"Image saved to {file_path}")
    else:
        print(f"Image already exists at {file_path}")

    messages = [HumanMessage(content="Add 3 and 4. Multiply the output by 2. Divide the output by 5")]
    result = react_graph.invoke({"messages": messages})
    for m in result['messages']:
        m.pretty_print()