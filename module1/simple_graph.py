import os
from random import random
from typing import TypedDict, Literal

from langgraph.constants import START, END
from langgraph.graph import StateGraph


# Agent State
class AgentState(TypedDict):
    graph_state: str


# Nodes
def node_1(state):
    print("___Node 01___")
    return {
        "graph_state": state["graph_state"] + " I am"
    }

def node_2(state):
    print("___Node 02___")
    return {
        "graph_state": state["graph_state"] + " happy!"
    }

def node_3(state):
    print("___Node 03___")
    return {
        "graph_state": state["graph_state"] + " sad!"
    }


# Edges
def decide_mood(state) -> Literal["node_2","node_3"]:
    user_input = state["graph_state"]
    print(f"Edge deciding the state: {user_input}")

    if random() > 0.5:
        return "node_2"
    else:
        return "node_3"


# Build the graph at module level so LangGraph CLI can discover it
builder = StateGraph(AgentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

graph = builder.compile()


if __name__ == "__main__":
    # Draw and save graph image
    os.makedirs("images", exist_ok=True)
    file_path = os.path.join("images", "simple_graph.png")

    if not os.path.exists(file_path):
        png_data = graph.get_graph().draw_mermaid_png()
        with open(file_path, "wb") as f:
            f.write(png_data)
        print(f"Image saved to {file_path}")
    else:
        print(f"Image already exists at {file_path}")

    result = graph.invoke({"graph_state": "Hello I'm jina."})
    print(result)