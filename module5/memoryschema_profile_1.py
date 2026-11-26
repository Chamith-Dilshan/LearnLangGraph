import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

load_dotenv()

"""
LangChain's chat model interface has a with_structured_output method to enforce structured output.
This is useful when we want to enforce that the output conforms to a schema, and it parses the 
output for us.

But when the schema become more complex, with_structured_output method may not be sufficient and 
may give you unexpected results.
In addition, updating even simple schemas can pose challenges.
Consider our chatbot.
We regenerated the profile schema from scratch each time we chose to save a new memory.
This is inefficient, potentially wasting model tokens if the schema contains a lot of information to 
re-generate each time.

Worse, we may loose information when regenerating the profile from scratch.
In such cases, we can use a library like Trustcall to avoid 
these issues.

"""


class UserProfile(TypedDict):
    """User profile schema with typed fields"""

    user_name: str  # The user's preferred name
    interests: list[str]  # A list of the user's interests


# TypedDict instance
user_profile: UserProfile = {
    "user_name": "Lance",
    "interests": ["biking", "technology", "coffee"],
}

# Initialize the in-memory store
in_memory_store = InMemoryStore()

# Namespace for the memory to save
user_id = "1"
namespace_for_memory = (user_id, "memory")

# Save a memory to namespace as key and value
key = "user_profile"
value = user_profile
in_memory_store.put(namespace_for_memory, key, value)


llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="openai/gpt-oss-20b",
    temperature=0,
)

# Bind schema to model
model_with_structure = llm.with_structured_output(UserProfile)

# Chatbot instruction
MODEL_SYSTEM_MESSAGE = """You are a helpful assistant with memory that provides information about the user. 
If you have memory for this user, use it to personalize your responses.
Here is the memory (it may be empty): {memory}"""

# Create new memory from the chat history and any existing memory
CREATE_MEMORY_INSTRUCTION = """Create or update a user profile memory based on the user's chat history. 
This will be saved for long-term memory. If there is an existing memory, simply update it. 
Here is the existing memory (it may be empty): {memory}"""


def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Load memory from the store and use it to personalize the chatbot's response."""

    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Retrieve memory from the store
    namespace = ("memory", user_id)
    existing_memory = store.get(namespace, "user_memory")

    # Format the memories for the system prompt
    if existing_memory and existing_memory.value:
        memory_dict = existing_memory.value
        formatted_memory = (
            f"Name: {memory_dict.get('user_name', 'Unknown')}\n"
            f"Interests: {', '.join(memory_dict.get('interests', []))}"
        )
    else:
        formatted_memory = None

    # Format the memory in the system prompt
    system_msg = MODEL_SYSTEM_MESSAGE.format(memory=formatted_memory)

    # Respond using memory as well as the chat history
    response = llm.invoke([SystemMessage(content=system_msg)] + state["messages"])

    return {"messages": response}


def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Reflect on the chat history and save a memory to the store."""

    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Retrieve existing memory from the store
    namespace = ("memory", user_id)
    existing_memory = store.get(namespace, "user_memory")

    # Format the memories for the system prompt
    if existing_memory and existing_memory.value:
        memory_dict = existing_memory.value
        formatted_memory = (
            f"Name: {memory_dict.get('user_name', 'Unknown')}\n"
            f"Interests: {', '.join(memory_dict.get('interests', []))}"
        )
    else:
        formatted_memory = None

    # Format the existing memory in the instruction
    system_msg = CREATE_MEMORY_INSTRUCTION.format(memory=formatted_memory)

    # Invoke the model to produce structured output that matches the schema
    new_memory = model_with_structure.invoke(
        [SystemMessage(content=system_msg)] + state["messages"]
    )

    # Overwrite the existing use profile memory
    key = "user_memory"
    store.put(namespace, key, new_memory)


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
    file_path = os.path.join("images", "memoryschema_profile_1.png")

    if not os.path.exists(file_path):
        png_data = graph.get_graph().draw_mermaid_png()
        with open(file_path, "wb") as f:
            f.write(png_data)
        print(f"Image saved to {file_path}")

    # Search
    print("Memory store data:")
    for m in in_memory_store.search(namespace_for_memory):
        print(m.dict())

    # Get the memory by namespace and key
    profile = in_memory_store.get(namespace_for_memory, "user_profile")
    if profile:
        print(f"\nProfile values: {profile.value}")

    # Invoke the model to produce structured output that matches the schema
    structured_output = model_with_structure.invoke(
        [HumanMessage("My name is Lance, I like to bike.")]
    )
    print(f"\nStructured output: {structured_output}")

    # We supply a thread ID for short-term (within-thread) memory
    # We supply a user ID for long-term (across-thread) memory
    config = {"configurable": {"thread_id": "1", "user_id": "1"}}

    # User input
    input_messages = [
        HumanMessage(
            content="Hi, my name is Lance and I like to bike around San Francisco and eat at bakeries."
        )
    ]

    # Run the graph
    for chunk in graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    # Namespace for the memory to save
    user_id = "1"
    namespace = ("memory", user_id)
    existing_memory = across_thread_memory.get(namespace, "user_memory")
    if existing_memory:
        print(f"\nExisting memory: {existing_memory.value}")
