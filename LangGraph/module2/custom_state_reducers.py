""" Sometimes predefined reducers are not enough.
For example, WE might need to define a custom reducer logic to combine
lists and handle cases where either or both of the inputs might be None.
"""
import os
from typing import TypedDict, Annotated

from langgraph.constants import START, END
from langgraph.graph import StateGraph


def custom_reduce_list(left: list | None, right: list | None) -> list:
    """Safely combine two lists, handling cases where either or both inputs might be None.

    Args:
        left (list | None): The first list to combine, or None.
        right (list | None): The second list to combine, or None.

    Returns:
        list: A new list containing all elements from both input lists.
               If an input is None, it's treated as an empty list.
    """

    if not left:
        left = []
    if not right:
        right = []

    return left + right

class AgentState(TypedDict):
    foo: Annotated[list[int], custom_reduce_list]

def node_1(state):
    print("---Node 1---")
    return {"foo": [2]}

# Build graph
builder = StateGraph(AgentState)
builder.add_node("node_1", node_1)

# Logic
builder.add_edge(START, "node_1")
builder.add_edge("node_1", END)

# Add
graph = builder.compile()

if __name__ == "__main__":
    # Draw and save graph image
    os.makedirs("images", exist_ok=True)
    file_path = os.path.join("images", "custom_state_reducer.png")

    if not os.path.exists(file_path):
        png_data = graph.get_graph().draw_mermaid_png()
        with open(file_path, "wb") as f:
            f.write(png_data)
        print(f"Image saved to {file_path}")
    else:
        print(f"Image already exists at {file_path}")

    custom_reducer_graph_result = graph.invoke({"foo": None})
    print(custom_reducer_graph_result)