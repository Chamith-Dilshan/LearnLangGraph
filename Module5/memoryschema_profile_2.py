import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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
Using TrustCall

We use create_extractor, passing in the model as well as our schema as a tool.
With TrustCall, can supply supply the schema in various ways.

For example, we can pass a JSON object / Python dictionary or Pydantic model.
Under the hood, TrustCall uses tool calling to produce structured output from an input list of messages.

To force Trustcall to produce structured output, we can include the schema name in the tool_choice argument.
We can invoke the extractor with the above conversation.

How to update the schema without rebuilding again.

The central idea is that it prompts the model to produce a JSON Patch to update only the relevant parts of the schema.
This is less error-prone than naively overwriting the entire schema.
It's also more efficient since the model only needs to generate the parts of the schema that have changed.
We can save the existing schema as a dict.

We can use model_dump() to serialize a Pydantic model instance into a dict.
We pass it to the "existing" argument along with the schema name, UserProfile.
"""

llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="qwen/qwen3.8-27b",
    temperature=0,
)


# Schema
class UserProfile(BaseModel):
    """Profile of a user"""

    user_name: str = Field(description="The user's preferred name")
    user_location: str = Field(description="The user's location")
    interests: list = Field(description="A list of the user's interests")


# Create the extractor
trustcall_extractor = create_extractor(
    llm,
    tools=[UserProfile],
    tool_choice="UserProfile",  # Enforces use of the UserProfile tool
)

# Chatbot instruction
MODEL_SYSTEM_MESSAGE = """You are a helpful assistant with memory that provides information about the user. 
If you have memory for this user, use it to personalize your responses.
Here is the memory (it may be empty): {memory}"""

# Extraction instruction
TRUSTCALL_INSTRUCTION = """Create or update the memory (JSON doc) to incorporate information from the following conversation:"""


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
            f"Location: {memory_dict.get('user_location', 'Unknown')}\n"
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

    # Get the profile as the value from the list, and convert it to a JSON doc
    existing_profile = (
        {"UserProfile": existing_memory.value} if existing_memory else None
    )

    # Invoke the extractor
    result = trustcall_extractor.invoke(
        {
            "messages": [SystemMessage(content=TRUSTCALL_INSTRUCTION)]
            + state["messages"],
            "existing": existing_profile,
        }
    )

    # Get the updated profile as a JSON object
    updated_profile = result["responses"][0].model_dump()

    # Save the updated profile
    key = "user_memory"
    store.put(namespace, key, updated_profile)


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
    file_path = os.path.join("images", "memoryschema_profile_2.png")

    if not os.path.exists(file_path):
        png_data = graph.get_graph().draw_mermaid_png()
        with open(file_path, "wb") as f:
            f.write(png_data)
        print(f"Image saved to {file_path}")

    # We supply a thread ID for short-term (within-thread) memory
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
    namespace = ("memory", user_id)
    existing_memory = across_thread_memory.get(namespace, "user_memory")
    if existing_memory:
        print(f"Existing memory: {existing_memory.dict()}")

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
