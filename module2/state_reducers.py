"""
 When you need your two or more nodes in the graph to run in parallel, which means they run in the same
 step of the graph. They both attempt to overwrite the state within the same step. This will throw
 InvalidUpdateError.

 Reducers give us a general way to address this problem.
 They specify how to perform updates.
 We can use the Annotated type to specify a reducer function.

"""
import os
from operator import add
from typing import TypedDict, Annotated

from langgraph.constants import START, END
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    """ in this case let's append the value returned from each node rather than overwriting them.
    We just need a reducer that can perform this: operator.add is a function from Python's built-in operator module.
    When operator.add is applied to lists, it performs list concatenation."""

    foo: Annotated[list[int], add]

def node_1(state):
    print("---Node 1---")
    return {"foo": [state['foo'][-1] + 1]}

def node_2(state):
    print("---Node 2---")
    return {"foo": [state['foo'][-1] + 1]}

def node_3(state):
    print("---Node 3---")
    return {"foo": [state['foo'][-1] + 1]}


# Build graph
builder = StateGraph(AgentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Logic
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_1", "node_3")
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# Add
graph = builder.compile()


if __name__ == "__main__":
    # Draw and save graph image
    os.makedirs("images", exist_ok=True)
    file_path = os.path.join("images", "state_reducer.png")

    if not os.path.exists(file_path):
        png_data = graph.get_graph().draw_mermaid_png()
        with open(file_path, "wb") as f:
            f.write(png_data)
        print(f"Image saved to {file_path}")
    else:
        print(f"Image already exists at {file_path}")

    reducer_graph_result = graph.invoke({"foo": [1]})
    print(reducer_graph_result)