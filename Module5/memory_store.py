import os
import uuid
from pprint import pprint

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
When storing objects (e.g., memories) in the Store, we provide:

1. The namespace for the object, a tuple (similar to directories)
2. The object key (similar to filenames)
3. The object value (similar to file contents)
"""

# We use the put method to save an object to the store by namespace and key.
in_memory_store = InMemoryStore()

# Namespace for the memory to save
user_id = "1"
namespace_for_memory = (user_id, "memories")

# Save a memory to namespace as key and value
key = str(uuid.uuid4())

# The value needs to be a dictionary
value = {"food_preference": "I like pizza"}

# Save the memory
in_memory_store.put(namespace_for_memory, key, value)

# We use search to retrieve objects from the store by namespace.
memories = in_memory_store.search(namespace_for_memory)

"""
Chatbot with long-term memory
two types of memory:

1. Short-term (within-thread) memory: Chatbot can persist conversational history and / or allow interruptions in a chat session.
2. Long-term (cross-thread) memory: Chatbot can remember information about a specific user across all chat sessions.

Short term memory ->
 For short-term memory, we'll use a checkpointer.
  # They write the graph state at each step to a thread.
  # They persist the chat history in the thread.
  # They allow the graph to be interrupted and / or resumed from any step in the thread.

Long term memory ->
 for long-term memory, we'll use the LangGraph Store.
    # You can preserve the memory across different chat sessions/threads.
    # You can use make the agent update the memory immidiately after each response or use a background job to update
      it on intervals.

Here the chat history will be saved to short-term memory using the checkpointer.
The chatbot will reflect on the chat history.
It will then create and save a memory to the LangGraph Store.
This memory is accessible in future chat sessions to personalize the chatbot's responses.

When we interact with the chatbot, we supply two things:

1. Short-term (within-thread) memory: A thread ID for persisting the chat history.
2. Long-term (cross-thread) memory: A user ID to namespace long-term memories to the user.
  
"""

llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="openai/gpt-oss-20b",
    temperature=0,
)

# Chatbot instruction
MODEL_SYSTEM_MESSAGE = """You are a helpful assistant with memory that provides information about the user. 
If you have memory for this user, use it to personalize your responses.
Here is the memory (it may be empty): {memory}"""

# Create new memory from the chat history and any existing memory
CREATE_MEMORY_INSTRUCTION = """"You are collecting information about the user to personalize your responses.

CURRENT USER INFORMATION:
{memory}

INSTRUCTIONS:
1. Review the chat history below carefully
2. Identify new information about the user, such as:
   - Personal details (name, location)
   - Preferences (likes, dislikes)
   - Interests and hobbies
   - Past experiences
   - Goals or future plans
3. Merge any new information with existing memory
4. Format the memory as a clear, bulleted list
5. If new information conflicts with existing memory, keep the most recent version

Remember: Only include factual information directly stated by the user. Do not make assumptions or inferences.

Based on the chat history below, please update the user information:"""


def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Load memory from the store and use it to personalize the chatbot's response."""

    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Retrieve memory from the store
    namespace = ("memory", user_id)
    key = "user_memory"
    existing_memory = store.get(namespace, key)

    # Extract the actual memory content if it exists and add a prefix
    if existing_memory:
        # Value is a dictionary with a memory key
        existing_memory_content = existing_memory.value.get("memory")
    else:
        existing_memory_content = "No existing memory found."

    # Format the memory in the system prompt
    system_msg = MODEL_SYSTEM_MESSAGE.format(memory=existing_memory_content)

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

    # Extract the memory
    if existing_memory:
        existing_memory_content = existing_memory.value.get("memory")
    else:
        existing_memory_content = "No existing memory found."

    # Format the memory in the system prompt
    system_msg = CREATE_MEMORY_INSTRUCTION.format(memory=existing_memory_content)
    new_memory = llm.invoke([SystemMessage(content=system_msg)] + state["messages"])

    # Overwrite the existing memory in the store
    key = "user_memory"

    # Write value as a dictionary with a memory key
    store.put(namespace, key, {"memory": new_memory.content})


# Define the graph
builder = StateGraph(MessagesState)

builder.add_node("call_model", call_model)
builder.add_node("write_memory", write_memory)

builder.add_edge(START, "call_model")
builder.add_edge("call_model", "write_memory")
builder.add_edge("write_memory", END)

# Checkpointer for short-term (within-thread) memory
within_thread_memory = MemorySaver()

# Store for long-term (across-thread) memory
across_thread_memory = InMemoryStore()

# Compile the graph with the checkpointer fir and store
graph = builder.compile(checkpointer=within_thread_memory, store=across_thread_memory)

if __name__ == "__main__":
    # Metatdata
    print("Print First memory block from the list -> ")
    pprint(memories[0].dict())
    print("\n")

    # The key, value
    print("Print First memory block's key and value pair -> ")
    pprint((memories[0].key, memories[0].value))
    print("\n")

    # Get the memory by namespace and key
    memory = in_memory_store.get(namespace_for_memory, key)
    if memory is not None:  # store.get() returns Item | None
        print("Print memory block's metadata -> ")
        pprint(memory.dict())
        print("\n")

    os.makedirs("images", exist_ok=True)
    file_path = os.path.join("images", "memory_store.png")

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

    thread = {"configurable": {"thread_id": "1"}}
    state = graph.get_state(thread).values
    for m in state["messages"]:
        m.pretty_print()

    user_id = "1"
    namespace = ("memory", user_id)
    existing_memory = across_thread_memory.get(namespace, "user_memory")

    if existing_memory:
        print(f"\nExisting memory: {existing_memory.dict()}")

    # We supply a user ID for across-thread memory as well as a new thread ID
    config = {"configurable": {"thread_id": "2", "user_id": "1"}}

    # User input
    input_messages = [
        HumanMessage(content="Hi! Where would you recommend that I go biking?")
    ]

    # Run the graph
    for chunk in graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    # User input
    input_messages = [
        HumanMessage(
            content="Great, are there any bakeries nearby that I can check out? I like a croissant after biking."
        )
    ]

    # Run the graph
    for chunk in graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()
