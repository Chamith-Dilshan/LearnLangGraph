""" Sometimes we may want to use different input / output schemas for the graph. While the input schema has
  different values and The output schema might only contain a single relevant output key."""
import os
from typing import TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph

""" 
Private State

This is useful for anything needed as part of the intermediate working logic of 
the graph, but not relevant for the overall graph input or output.
"""

class AgentState(TypedDict):
    foo: int

class PrivateState(TypedDict):
    baz: int

def node_1(state: AgentState) -> PrivateState:
    return {"baz": state['foo'] + 1}

def node_2(state: PrivateState) -> AgentState:
    return {"foo": state['baz'] + 1}

# Build graph
builder1 = StateGraph(AgentState)
builder1.add_node("node_1", node_1)
builder1.add_node("node_2", node_2)

# Logic
builder1.add_edge(START, "node_1")
builder1.add_edge("node_1", "node_2")
builder1.add_edge("node_2", END)

# Add
graph1 = builder1.compile()

""" 
Input / Output Schema

Define explicit input and output schemas for a graph.
In these cases, we often define an "internal" schema that contains all keys relevant to graph operations.
But we use specific input and output schemas to constrain the input and output.
"""

class InputState(TypedDict):
    question: str

class OutputState(TypedDict):
    answer: str

class OverallState(TypedDict):
    question: str
    answer: str
    notes: str

def thinking_node(state: InputState):
    return {"answer": "bye", "notes": "... his is name is Lance"}

def answer_node(state: OverallState) -> OutputState:
    return {"answer": "bye Lance"}

builder2 = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
builder2.add_node("answer_node", answer_node)
builder2.add_node("thinking_node", thinking_node)

builder2.add_edge(START, "thinking_node")
builder2.add_edge("thinking_node", "answer_node")
builder2.add_edge("answer_node", END)

graph2 = builder2.compile()


if __name__ == "__main__":
    # Draw and save graph image
    os.makedirs("images", exist_ok=True)
    file_path1 = os.path.join("images", "private_state.png")
    file_path2 = os.path.join("images", "Input_output_state.png")

    if not os.path.exists(file_path1 and file_path2):
        png_data1 = graph1.get_graph().draw_mermaid_png()
        png_data2 = graph2.get_graph().draw_mermaid_png()
        with open(file_path1, "wb") as f:
            f.write(png_data1)
        with open(file_path2, "wb") as f:
            f.write(png_data2)
        print(f"Image saved to {file_path1} and {file_path2}")
    else:
        print(f"Image already exists at {file_path1} and {file_path1}")

    private_state_graph_result = graph1.invoke({"foo": 1})
    print("Private State : ",private_state_graph_result)

    input_output_state_graph_result = graph2.invoke({"question": "hello"})
    print("Input Output State : ",input_output_state_graph_result)