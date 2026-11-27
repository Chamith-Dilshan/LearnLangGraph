import operator
import os
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.errors import InvalidUpdateError
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    state: Annotated[list[str], operator.add]


def sorting_reducer(left, right):
    """Combines and sorts the values in a list"""
    if not isinstance(left, list):
        left = [left]

    if not isinstance(right, list):
        right = [right]

    return sorted(left + right, reverse=False)


class AgentState2(TypedDict):
    state: Annotated[list[str], sorting_reducer]


class ReturnNodeValue:
    def __init__(self, value: str):
        self._value = value

    def __call__(self, state):
        print(f"Adding {self._value} to {state['state']}")
        return {"state": [self._value]}


# Build Graph 1
builder = StateGraph(AgentState)
builder.add_node("a", ReturnNodeValue("I'm A"))
builder.add_node("b", ReturnNodeValue("I'm B"))
builder.add_node("c", ReturnNodeValue("I'm C"))
builder.add_node("d", ReturnNodeValue("I'm D"))

builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("a", "c")
builder.add_edge("b", "d")
builder.add_edge("c", "d")
builder.add_edge("d", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


# Build Graph 2
builder2 = StateGraph(AgentState2)
builder2.add_node("a", ReturnNodeValue("I'm A"))
builder2.add_node("b", ReturnNodeValue("I'm B"))
builder2.add_node("b2", ReturnNodeValue("I'm B2"))
builder2.add_node("c", ReturnNodeValue("I'm C"))
builder2.add_node("d", ReturnNodeValue("I'm D"))

builder2.add_edge(START, "a")
builder2.add_edge("a", "b")
builder2.add_edge("a", "c")
builder2.add_edge("b", "b2")
builder2.add_edge("b2", "d")
builder2.add_edge("c", "d")
builder2.add_edge("d", END)

memory2 = MemorySaver()
graph2 = builder2.compile(checkpointer=memory2)


if __name__ == "__main__":
    os.makedirs("images", exist_ok=True)
    file_path = os.path.join("images", "parallelization.png")
    file_path2 = os.path.join("images", "parallelization2.png")

    if not os.path.exists(file_path):
        png_data = graph.get_graph().draw_mermaid_png()
        with open(file_path, "wb") as f:
            f.write(png_data)
        print(f"Image saved to {file_path}")

    if not os.path.exists(file_path2):
        png_data2 = graph2.get_graph().draw_mermaid_png()
        with open(file_path2, "wb") as f:
            f.write(png_data2)
        print(f"Image saved to {file_path2}")

    # Invoke Graph 1 with its own thread ID
    thread_1 = {"configurable": {"thread_id": "thread-graph-1"}}
    try:
        graph.invoke({"state": []}, config=thread_1)
    except InvalidUpdateError as e:
        print(f"An error occurred: {e}")

    print(f"AgentState 1: {graph.get_state(thread_1)}\n")

    # Invoke Graph 2 with a separate thread ID and state schema
    thread_2 = {"configurable": {"thread_id": "thread-graph-2"}}
    try:
        graph2.invoke({"state": []}, config=thread_2)
    except InvalidUpdateError as e:
        print(f"An error occurred: {e}")

    print(f"AgentState 2: {graph2.get_state(thread_2)}\n")
