import os
import uuid

from dotenv import load_dotenv
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    merge_message_runs,
)
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field
from trustcall import create_extractor

load_dotenv()

"""
Sometimes we want to save memories to a collection rather than single profile.
Here we'll update our chatbot to save memories to a collection.
Here to show how to use Trustcall to update this collection.

Instead of storing user information in a fixed profile structure, we'll create 
a flexible collection schema to store memories about user interactions.
Each memory will be stored as a separate entry with a single content field 
for the main information we want to remember
This approach allows us to build an open-ended collection of memories that can grow 
and change as we learn more about the user.
"""

llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="qwen/qwen3.8-27b",
    temperature=0,
)

"""
Rule of thumb for trustcall: every Pydantic schema you pass to create_extractor should have a docstring, 
since each becomes a tool in the request. (This is also just good extraction hygiene — the description 
is what the LLM uses to decide when to call the tool.)
"""


# Memory schema
class Memory(BaseModel):
    """Store a single memory about the user.

    Use this tool to retain important information learned from the conversation,
    such as preferences, personal details, interests, and goals.
    """

    content: str = Field(
        description="The main content of the memory. For example: User expressed interest in learning about French."
    )


# Create the extractor
trustcall_extractor = create_extractor(
    llm,
    tools=[Memory],
    tool_choice="Memory",
    # This allows the extractor to insert new memories
    enable_inserts=True,
)

# Chatbot instruction
MODEL_SYSTEM_MESSAGE = """You are a helpful chatbot. You are designed to be a companion to a user. 
You have a long term memory which keeps track of information you learn about the user over time.
Current Memory (may include updated memories from this conversation): 
{memory}"""

# Trustcall instruction
TRUSTCALL_INSTRUCTION = """Reflect on following interaction. 
Use the provided tools to retain any necessary memories about the user. 
Use parallel tool calling to handle updates and insertions simultaneously:"""


def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Load memories from the store and use them to personalize the chatbot's response."""

    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Retrieve memory from the store
    namespace = ("memories", user_id)
    memories = store.search(namespace)

    # Format the memories for the system prompt
    info = "\n".join(f"- {mem.value['content']}" for mem in memories)
    system_msg = MODEL_SYSTEM_MESSAGE.format(memory=info)

    # Respond using memory as well as the chat history
    response = llm.invoke([SystemMessage(content=system_msg)] + state["messages"])

    return {"messages": response}


def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Reflect on the chat history and update the memory collection."""

    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Define the namespace for the memories
    namespace = ("memories", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name = "Memory"
    existing_memories = (
        [
            (existing_item.key, tool_name, existing_item.value)
            for existing_item in existing_items
        ]
        if existing_items
        else None
    )

    # Merge the chat history and the instruction
    updated_messages = list(
        merge_message_runs(
            [SystemMessage(content=TRUSTCALL_INSTRUCTION)] + state["messages"]
        )
    )

    # Invoke the extractor
    result = trustcall_extractor.invoke(
        {"messages": updated_messages, "existing": existing_memories}
    )

    # Save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(
            namespace,
            rmeta.get("json_doc_id", str(uuid.uuid4())),
            r.model_dump(mode="json"),
        )


# Define the graph
builder = StateGraph(MessagesState)

builder.add_node("call_model", call_model)
builder.add_node("write_memory", write_memory)

builder.add_edge(START, "call_model")
builder.add_edge("call_model", "write_memory")
builder.add_edge("write_memory", END)

# Store for long-term (across-thread) memory
across_thread_memory = InMemoryStore()

# Checkpointer for short-term (within-thread) memory
within_thread_memory = MemorySaver()

# Compile the graph with the checkpointer fir and store
graph = builder.compile(checkpointer=within_thread_memory, store=across_thread_memory)

if __name__ == "__main__":
    os.makedirs("images", exist_ok=True)
    file_path = os.path.join("images", "memoryschema_collection.png")

    if not os.path.exists(file_path):
        png_data = graph.get_graph().draw_mermaid_png()
        with open(file_path, "wb") as f:
            f.write(png_data)
        print(f"Image saved to {file_path}")

    # We supply a thread ID for short-term (within-thread) memory.
    # We supply a user ID for long-term (across-thread) memory
    config = {"configurable": {"thread_id": "1", "user_id": "1"}}

    # User input
    input_messages = [HumanMessage(content="Hi, my name is Lance")]

    # Run the graph
    for chunk in graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    # User input
    input_messages = [HumanMessage(content="I like to bike around San Francisco")]

    # Run the graph
    for chunk in graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    # Namespace for the memory to save
    user_id = "1"
    namespace = ("memories", user_id)
    memories = across_thread_memory.search(namespace)
    for m in memories:
        print(m.dict())

    # User input
    input_messages = [HumanMessage(content="I also enjoy going to bakeries")]

    # Run the graph
    for chunk in graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    # We supply a thread ID for short-term (within-thread) memory
    # We supply a user ID for long-term (across-thread) memory
    config = {"configurable": {"thread_id": "2", "user_id": "1"}}

    # User input
    input_messages = [HumanMessage(content="What bakeries do you recommend for me?")]

    # Run the graph
    for chunk in graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()
