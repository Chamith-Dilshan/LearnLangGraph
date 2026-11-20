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


# Graph
builder = StateGraph(MessagesState)

# Define Nodes
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))

# Define edges
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
builder.add_edge("agent", END)

memory = MemorySaver()
# graph = builder.compile(checkpointer=memory)
graph = builder.compile()


async def replaying_api(client, initial_input, thread):
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id="agent",
        input=initial_input,
        stream_mode="updates",
    ):
        if chunk.data:
            assisant_node = chunk.data.get("assistant", {}).get("messages", [])
            tool_node = chunk.data.get("tools", {}).get("messages", [])
            if assisant_node:
                print("-" * 20 + "Assistant Node" + "-" * 20)
                print(assisant_node[-1])
            elif tool_node:
                print("-" * 20 + "Tools Node" + "-" * 20)
                print(tool_node[-1])

    states = await client.threads.get_history(thread["thread_id"])
    to_replay = states[-2]
    print("current state:", to_replay)

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id="agent",
        input=None,
        stream_mode="updates",
        checkpoint_id=to_replay["checkpoint_id"],
    ):
        if chunk.data:
            assisant_node = chunk.data.get("assistant", {}).get("messages", [])
            tool_node = chunk.data.get("tools", {}).get("messages", [])
            if assisant_node:
                print("-" * 20 + "Assistant Node" + "-" * 20)
                print(assisant_node[-1])
            elif tool_node:
                print("-" * 20 + "Tools Node" + "-" * 20)
                print(tool_node[-1])


async def forking_api(client, initial_input, thread):
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id="agent",
        input=initial_input,
        stream_mode="updates",
    ):
        if chunk.data:
            assisant_node = chunk.data.get("assistant", {}).get("messages", [])
            tool_node = chunk.data.get("tools", {}).get("messages", [])
            if assisant_node:
                print("-" * 20 + "Assistant Node" + "-" * 20)
                print(assisant_node[-1])
            elif tool_node:
                print("-" * 20 + "Tools Node" + "-" * 20)
                print(tool_node[-1])

    states = await client.threads.get_history(thread["thread_id"])
    to_fork = states[-2]
    print("current state:", to_fork)

    forked_input = {
        "messages": HumanMessage(
            content="Multiply 3 and 3", id=to_fork["values"]["messages"][0]["id"]
        )
    }

    forked_config = await client.threads.update_state(
        thread["thread_id"], forked_input, checkpoint_id=to_fork["checkpoint_id"]
    )

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id="agent",
        input=None,
        stream_mode="updates",
        checkpoint_id=forked_config["checkpoint_id"],
    ):
        if chunk.data:
            assisant_node = chunk.data.get("assistant", {}).get("messages", [])
            tool_node = chunk.data.get("tools", {}).get("messages", [])
            if assisant_node:
                print("-" * 20 + "Assistant Node" + "-" * 20)
                print(assisant_node[-1])
            elif tool_node:
                print("-" * 20 + "Tools Node" + "-" * 20)
                print(tool_node[-1])


if __name__ == "__main__":
    # Draw and save graph image
    os.makedirs("images", exist_ok=True)
    file_path = os.path.join("images", "time_travel.png")

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

    # use get_state to look at the current state of our graph, given the thread_id
    current_state = graph.get_state(thread)
    print(f"current state of the agent: {current_state}")

    # Browse the state history of our agent.
    print("\nHow Browse the state history work...\n")

    # browse the state history of our agent.
    all_states = [s for s in graph.get_state_history(thread)]
    print(f"history of the state of the agent: {all_states}")

    # The first element is the current state, just as we got from get_state.
    print(f"first element of the state history: {all_states[-2]}")

    # Replaying
    print("\nHow Replaying work...\n")
    to_replay = all_states[-2]
    print(f"To replay state: {to_replay}")
    print(f"state value: {to_replay.values}")
    print(f"the next node to call: {to_replay.next}")
    print(
        f" We also get the config, which tells us the checkpoint_id as well as the thread_id : {to_replay.config}"
    )
    print(
        "To replay from here, we simply pass the config back to the agent!\nThe graph knows that this checkpoint has aleady been executed.\nIt just re-plays from this checkpoint!\n it won't re-execute the graph from that point"
    )

    for event in graph.stream(None, to_replay.config, stream_mode="values"):
        event["messages"][-1].pretty_print()

    # Forking
    print("\nHow Forking work...\n")
    print("Run from that same step, but with a different input")

    to_fork = all_states[-2]
    print(f"to fork values messages: {to_fork.values['messages']}")
    print(f"to fork config: {to_fork.config}")
    print(
        "Remember how our reducer on messages works:\n1.It will append, unless we supply a message ID.\n2.We supply the message ID to overwrite the message, rather than appending to state!"
    )

    fork_config = graph.update_state(
        to_fork.config,
        {
            "messages": [
                HumanMessage(
                    content="Multiply 5 and 3", id=to_fork.values["messages"][0].id
                )
            ]
        },
    )

    print("This creates a new, forked checkpoint.")
    print("But, the metadata - e.g., where to go next - is perserved!")

    all_states = [state for state in graph.get_state_history(thread)]
    print(f"updated message: {all_states[0].values['messages']}")

    for event in graph.stream(None, fork_config, stream_mode="values"):
        event["messages"][-1].pretty_print()

    # Working with LangGraph API
    client = get_client(url="http://127.0.0.1:2024")

    # Replaying
    # asyncio.run(replaying_api(client, initial_input, thread))

    # Forking
    # asyncio.run(forking_api(client, initial_input, thread))
