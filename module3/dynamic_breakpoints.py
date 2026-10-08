import os
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt


class State(TypedDict):
    input: str | HumanMessage


def step_1(state: State) -> State:
    print("---Step 1---")
    return state


def step_2(state: State) -> State:
    # Extract text content whether input is a string or HumanMessage
    text_content = (
        state["input"].content
        if isinstance(state["input"], HumanMessage)
        else state["input"]
    )

    if len(text_content) > 5:
        # Use interrupt() for dynamic human-in-the-loop pauses
        interrupt(f"Received input that is longer than 5 characters: {text_content}")

    print("---Step 2---")
    return state


def step_3(state: State) -> State:
    print("---Step 3---")
    return state


builder = StateGraph(State)

builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)

builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

if __name__ == "__main__":
    os.makedirs("images", exist_ok=True)
    file_path = os.path.join("images", "dynamic_breakpoints.png")

    if not os.path.exists(file_path):
        png_data = graph.get_graph().draw_mermaid_png()
        with open(file_path, "wb") as f:
            f.write(png_data)
        print(f"Image saved to {file_path}")
    else:
        print(f"Image already exists at {file_path}")

    initial_input = {"input": HumanMessage(content="hello world")}
    thread = {"configurable": {"thread_id": "1"}}

    for event in graph.stream(initial_input, thread, stream_mode="values"):
        print(event)

    for event in graph.stream(None, thread, stream_mode="values"):
        print(event)

    graph.update_state(
        thread,
        {"input": "hi"},
    )

    for event in graph.stream(None, thread, stream_mode="values"):
        print(event)
