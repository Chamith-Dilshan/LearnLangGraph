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

from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, RemoveMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph, add_messages, MessagesState


class AgentState(TypedDict):
    """ in this case let's append the value returned from each node rather than overwriting them.
    We just need a reducer that can perform this: operator.add is a function from Python's built-in operator module.
    When operator.add is applied to lists, it performs list concatenation."""

    foo: Annotated[list[int], add]


""" MessagesState is a useful shortcut if you want to work with messages.

        MessagesState has a built-in messages key
        It also has a built-in add_messages reducer for this key
"""
# Define a custom TypedDict that includes a list of messages with add_messages reducer
class CustomMessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    added_key_1: str
    added_key_2: str
    # etc

# Use MessagesState, which includes the messages key with add_messages reducer
class ExtendedMessagesState(MessagesState):
    # Add any keys needed beyond messages, which is pre-built
    added_key_1: str
    added_key_2: str
    # etc

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

    print("\n")
    print("//" * 40)
    # Message Removal
    print("\nMessage Re-writing")

    """
    If we pass a message with the same ID as an existing one in our messages list, it will get overwritten!
    """
    # Initial state
    initial_messages = [AIMessage(content="Hello! How can I assist you?", name="Model", id="1"),
                        HumanMessage(content="I'm looking for information on marine biology.", name="Lance", id="2")
                        ]

    # New message to add
    new_message = HumanMessage(content="I'm looking for information on whales, specifically", name="Lance", id="2")

    # Test
    add_messages(initial_messages, new_message)
    current_messages_list = add_messages(initial_messages, new_message)
    print("Current messages :",current_messages_list)

    print("\n")
    print("//" * 40)
    # Message Removal
    print("\nMessage Removal")
    # Message list
    messages = [AIMessage("Hi.", name="Bot", id="1")]
    messages.append(HumanMessage("Hi.", name="Lance", id="2"))
    messages.append(AIMessage("So you said you were researching ocean mammals?", name="Bot", id="3"))
    messages.append(
        HumanMessage("Yes, I know about whales. But what others should I learn about?", name="Lance", id="4"))

    # Isolate messages to delete
    delete_messages = [RemoveMessage(id=m.id) for m in messages[:-2]]
    print("Deleted messages :",delete_messages)
    current_messages = add_messages(messages, delete_messages)
    print("Current messages :",current_messages)
