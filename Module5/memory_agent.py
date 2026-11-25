import os
import uuid
from datetime import UTC, datetime
from typing import Literal, LiteralString, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
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
This is a todolist management agent. Agent uses all three types of memories:
 - semantic memory -> Facts (e.g. Facts about user)
 - episodic memory -> Memories (e.g. Past agent actions)
 - procedural memory -> Instructions (e.g. Skills, Agent's system prompts)
"""


class Memory(BaseModel):
    """User specific memory"""

    content: str = Field(
        description="The main content of the memory. For example: User expressed interest in learning about French."
    )


class MemoryCollection(BaseModel):
    memories: list[Memory] = Field(description="A list of memories about the user.")


"""
let's add Visibility into Trustcall updates, so we can give a
nice update to the user

We can add a listener to the Trustcall extractor.
This will pass runs from the extractor's execution to a class, Spy, that we will define.
Our Spy class will extract information about what tool calls were made by Trustcall.
"""


# Inspect the tool calls made by Trustcall
class Spy:
    def __init__(self):
        self.called_tools = []

    def __call__(self, run):
        # Collect information about the tool calls made by the extractor.
        q = [run]
        while q:
            r = q.pop()
            if r.child_runs:
                q.extend(r.child_runs)
            if r.run_type == "chat_model":
                self.called_tools.append(
                    r.outputs["generations"][0][0]["message"]["kwargs"]["tool_calls"]
                )


# Initialize the spy
spy = Spy()

# Initialize the model
llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="qwen/qwen3.8-27b",
    temperature=0,
)

# Create the extractor
trustcall_extractor = create_extractor(
    llm,
    tools=[Memory],
    tool_choice="Memory",
    enable_inserts=True,
)

# Add the spy as a listener
trustcall_extractor_see_all_tool_calls = trustcall_extractor.with_listeners(on_end=spy)

# Instruction
instruction = """Extract memories from the following conversation:"""

""" 
There are many different agent architectures to choose from.
Here, we'll implement something simple, a ReAct agent.
This agent will be a helpful companion for creating and managing a ToDo list.
This agent can make a decision to update three types of long-term memory:

(a) Create or update a user profile with general user information
(b) Add or update items in a ToDo list collection
(c) Update its own instructions on how to update items to the ToDo list
"""


# Update memory tool
class UpdateMemory(TypedDict):
    """Decision on what memory type to update"""

    update_type: Literal["user", "todo", "instructions"]


# User profile schema
class Profile(BaseModel):
    """This is the profile of the user you are chatting with"""

    name: str | None = Field(description="The user's name", default=None)
    location: str | None = Field(description="The user's location", default=None)
    job: str | None = Field(description="The user's job", default=None)
    connections: list[str] = Field(
        description="Personal connection of the user, such as family members, friends, or coworkers",
        default_factory=list,
    )
    interests: list[str] = Field(
        description="Interests that the user has", default_factory=list
    )


class ToDo(BaseModel):
    """A model representing a to-do items status."""

    task: str = Field(description="The task to be completed.")
    time_to_complete: int | None = Field(
        description="Estimated time to complete the task (minutes)."
    )
    deadline: datetime | None = Field(
        description="When the task needs to be completed by (if applicable)",
        default=None,
    )
    solutions: list[str] = Field(
        description="List of specific, actionable solutions (e.g., specific ideas, service providers, or concrete options relevant to completing the task)",
        min_length=1,
        default_factory=list,
    )
    status: Literal["not started", "in progress", "done", "archived"] = Field(
        description="Current status of the task", default="not started"
    )


# Create the Trustcall extractor for updating the user profile
profile_extractor = create_extractor(
    llm,
    tools=[Profile],
    tool_choice="Profile",
)

# Chatbot instruction for choosing what to update and what tools to call
MODEL_SYSTEM_MESSAGE = """You are a helpful chatbot. 

You are designed to be a companion to a user, helping them keep track of their ToDo list.

You have a long term memory which keeps track of three things:
1. The user's profile (general information about them) 
2. The user's ToDo list
3. General instructions for updating the ToDo list

Here is the current User Profile (may be empty if no information has been collected yet):
<user_profile>
{user_profile}
</user_profile>

Here is the current ToDo List (may be empty if no tasks have been added yet):
<todo>
{todo}
</todo>

Here are the current user-specified preferences for updating the ToDo list (may be empty if no preferences have been specified yet):
<instructions>
{instructions}
</instructions>

Here are your instructions for reasoning about the user's messages:

1. Reason carefully about the user's messages as presented below. 

2. Decide whether any of the your long-term memory should be updated:
- If personal information was provided about the user, update the user's profile by calling UpdateMemory tool with type `user`
- If tasks are mentioned, update the ToDo list by calling UpdateMemory tool with type `todo`
- If the user has specified preferences for how to update the ToDo list, update the instructions by calling UpdateMemory tool with type `instructions`

3. Tell the user that you have updated your memory, if appropriate:
- Do not tell the user you have updated the user's profile
- Tell the user them when you update the todo list
- Do not tell the user that you have updated instructions

4. Err on the side of updating the todo list. No need to ask for explicit permission.

5. Respond naturally to user user after a tool call was made to save memories, or if no tool call was made."""

# Trustcall instruction
TRUSTCALL_INSTRUCTION = """Reflect on following interaction. 

Use the provided tools to retain any necessary memories about the user. 

Use parallel tool calling to handle updates and insertions simultaneously.

System Time: {time}"""

# Instructions for updating the ToDo list
CREATE_INSTRUCTIONS = """Reflect on the following interaction.

Based on this interaction, update your instructions for how to update ToDo list items. 

Use any feedback from the user to update how they like to have items added, etc.

Your current instructions are:

<current_instructions>
{current_instructions}
</current_instructions>"""


def extract_tool_info(tool_calls, schema_name="Memory"):
    """Extract information from tool calls for both patches and new memories safely.

    Args:
        tool_calls: List of tool calls from the model
        schema_name: Name of the schema tool (e.g., "Memory", "ToDo", "Profile")
    """

    # Initialize list of changes
    changes = []

    for call_group in tool_calls:
        for call in call_group:
            call_name = call.get("name")
            args = call.get("args", {})
            if call_name == "PatchDoc":
                patches = args.get("patches", [])
                value = patches[0].get("value") if patches else args
                changes.append(
                    {
                        "type": "update",
                        "doc_id": args.get("json_doc_id", "unknown"),
                        "planned_edits": args.get("planned_edits", ""),
                        "value": value,
                    }
                )
            elif call_name == schema_name or call_name:
                changes.append({"type": "new", "value": args})

    # Format results as a single string
    result_parts = []
    for change in changes:
        if change["type"] == "update":
            result_parts.append(
                f"Document {change['doc_id']} updated:\n"
                f"Plan: {change['planned_edits']}\n"
                f"Added content: {change['value']}"
            )
        else:
            result_parts.append(
                f"New {schema_name} created:\nContent: {change['value']}"
            )

    return (
        "\n\n".join(result_parts)
        if result_parts
        else f"Updated {schema_name} successfully."
    )


# Node definitions
def task_todo_agent(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Load memories from the store and use them to personalize the chatbot's response."""

    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Retrieve profile memory from the store
    namespace = ("profile", user_id)
    memories = store.search(namespace)
    if memories:
        user_profile = memories[0].value
    else:
        user_profile = None

    # Retrieve task memory from the store
    namespace = ("todo", user_id)
    memories = store.search(namespace)
    todo = "\n".join(f"{mem.value}" for mem in memories)

    # Retrieve custom instructions
    namespace = ("instructions", user_id)
    memories = store.search(namespace)
    if memories:
        instructions = memories[0].value
    else:
        instructions = ""

    system_msg = MODEL_SYSTEM_MESSAGE.format(
        user_profile=user_profile, todo=todo, instructions=instructions
    )

    # Respond using memory as well as the chat history
    response = llm.bind_tools([UpdateMemory], parallel_tool_calls=False).invoke(
        [SystemMessage(content=system_msg)] + state["messages"]
    )

    return {"messages": [response]}


def update_profile(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Reflect on the chat history and update the memory collection."""

    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Define the namespace for the memories
    namespace = ("profile", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name = "Profile"
    existing_memories = (
        [
            (existing_item.key, tool_name, existing_item.value)
            for existing_item in existing_items
        ]
        if existing_items
        else None
    )

    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(
        time=datetime.now(UTC).isoformat()
    )
    updated_messages = list(
        merge_message_runs(
            [SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)]
            + state["messages"][:-1]
        )
    )

    # Invoke the extractor
    result = profile_extractor.invoke(
        {"messages": updated_messages, "existing": existing_memories}
    )

    # Save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(
            namespace,
            rmeta.get("json_doc_id", str(uuid.uuid4())),
            r.model_dump(mode="json"),
        )
    last_message = state["messages"][-1]
    tool_call_id = getattr(last_message, "tool_calls", [{}])[0].get("id", "default_id")

    return {
        "messages": [
            ToolMessage(
                content="updated profile",
                tool_call_id=tool_call_id,
            )
        ]
    }


def update_todos(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Reflect on the chat history and update the memory collection."""

    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Define the namespace for the memories
    namespace = ("todo", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name = "ToDo"
    existing_memories = (
        [
            (existing_item.key, tool_name, existing_item.value)
            for existing_item in existing_items
        ]
        if existing_items
        else None
    )

    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(
        time=datetime.now(UTC).isoformat()
    )
    updated_messages = list(
        merge_message_runs(
            [SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)]
            + state["messages"][:-1]
        )
    )

    # Initialize the spy for visibility into the tool calls made by Trustcall
    spy = Spy()

    # Create the Trustcall extractor for updating the ToDo list
    todo_extractor = create_extractor(
        llm, tools=[ToDo], tool_choice=tool_name, enable_inserts=True
    ).with_listeners(on_end=spy)

    # Invoke the extractor
    result = todo_extractor.invoke(
        {"messages": updated_messages, "existing": existing_memories}
    )

    # Save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(
            namespace,
            rmeta.get("json_doc_id", str(uuid.uuid4())),
            r.model_dump(mode="json"),
        )

    # Respond to the tool call made in task_todo_agent, confirming the update
    # Safely retrieve tool calls from the last message
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []
    tool_call_id = tool_calls[0].get("id", "default_id") if tool_calls else "default_id"

    # Extract the changes made by Trustcall and add to the ToolMessage returned
    todo_update_msg = extract_tool_info(spy.called_tools, tool_name)
    return {
        "messages": [
            ToolMessage(
                content=todo_update_msg,
                tool_call_id=tool_call_id,
            )
        ]
    }


def update_instructions(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Reflect on the chat history and update the memory collection."""

    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    namespace = ("instructions", user_id)

    existing_memory = store.get(namespace, "user_instructions")

    # Format the memory in the system prompt
    system_msg = CREATE_INSTRUCTIONS.format(
        current_instructions=existing_memory.value if existing_memory else None
    )
    new_memory = llm.invoke(
        [SystemMessage(content=system_msg)]
        + state["messages"][:-1]
        + [
            HumanMessage(
                content="Please update the instructions based on the conversation"
            )
        ]
    )

    # Overwrite the existing memory in the store
    key = "user_instructions"
    store.put(namespace, key, {"memory": new_memory.content})
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []
    tool_call_id = tool_calls[0].get("id", "default_id") if tool_calls else "default_id"

    return {
        "messages": [
            {
                "role": "tool",
                "content": "updated instructions",
                "tool_call_id": tool_call_id,
            }
        ]
    }


# Conditional edge
def route_message(
    state: MessagesState, config: RunnableConfig, store: BaseStore
) -> Literal["update_todos", "update_instructions", "update_profile", "__end__"]:
    """Reflect on the memories and chat history to decide whether to update the memory collection."""
    message = state["messages"][-1]
    tool_calls = getattr(message, "tool_calls", None) or []
    if len(tool_calls) == 0:
        return "__end__"
    else:
        tool_call = tool_calls[0]
        update_type = tool_call.get("args", {}).get("update_type")
        if update_type == "user":
            return "update_profile"
        elif update_type == "todo":
            return "update_todos"
        elif update_type == "instructions":
            return "update_instructions"
        else:
            raise ValueError(f"Unknown update_type: {update_type}")


# Create the graph + all nodes
builder = StateGraph(MessagesState)

# Define the flow of the memory extraction process
builder.add_node(task_todo_agent)
builder.add_node(update_todos)
builder.add_node(update_profile)
builder.add_node(update_instructions)

builder.add_edge(START, "task_todo_agent")
builder.add_conditional_edges("task_todo_agent", route_message)
builder.add_edge("update_todos", "task_todo_agent")
builder.add_edge("update_profile", "task_todo_agent")
builder.add_edge("update_instructions", "task_todo_agent")

# Store for long-term (across-thread) memory
across_thread_memory = InMemoryStore()

# Checkpointer for short-term (within-thread) memory
within_thread_memory = MemorySaver()

# We compile the graph with the checkpointer and store
graph = builder.compile(checkpointer=within_thread_memory, store=across_thread_memory)

if __name__ == "__main__":
    # Conversation
    conversation = [
        HumanMessage(content="Hi, I'm Lance."),
        AIMessage(content="Nice to meet you, Lance."),
        HumanMessage(content="This morning I had a nice bike ride in San Francisco."),
    ]

    # Invoke the extractor
    result = trustcall_extractor.invoke(
        {"messages": [SystemMessage(content=instruction)] + conversation}
    )

    print(f"res: {result}")
    print("//" * 20, "\n")

    os.makedirs("images", exist_ok=True)
    file_path = os.path.join("images", "memory_agent.png")

    if not os.path.exists(file_path):
        png_data = graph.get_graph().draw_mermaid_png()
        with open(file_path, "wb") as f:
            f.write(png_data)
        print(f"Image saved to {file_path}")

    # We supply a thread ID for short-term (within-thread) memory
    # We supply a user ID for long-term (across-thread) memory
    config = {"configurable": {"thread_id": "1", "user_id": "Lance"}}

    # User input to create a profile memory
    input_messages = [
        HumanMessage(
            content="My name is Lance. I live in SF with my wife. I have a 1 year old daughter."
        )
    ]

    # Run the graph
    for chunk in graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    # User input for a ToDo
    input_messages = [
        HumanMessage(content="My wife asked me to book swim lessons for the baby.")
    ]

    # Run the graph
    for chunk in graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    # User input to update instructions for creating ToDos
    input_messages = [
        HumanMessage(
            content="When creating or updating ToDo items, include specific local businesses / vendors."
        )
    ]

    # Run the graph
    for chunk in graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    # Check for updated instructions
    user_id = "Lance"

    # Search
    for memory in across_thread_memory.search(("instructions", user_id)):
        print(memory.value)

    # User input for a ToDo
    input_messages = [
        HumanMessage(content="I need to fix the jammed electric Yale lock on the door.")
    ]

    # Run the graph
    for chunk in graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    # Namespace for the memory to save
    user_id = "Lance"

    # Search
    for memory in across_thread_memory.search(("todo", user_id)):
        print(memory.value)

    # User input to update an existing ToDo
    input_messages = [
        HumanMessage(
            content="For the swim lessons, I need to get that done by end of November."
        )
    ]

    # Run the graph
    for chunk in graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    # User input for a ToDo
    input_messages = [
        HumanMessage(content="Need to call back City Toyota to schedule car service.")
    ]

    # Run the graph
    for chunk in graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    # Namespace for the memory to save
    user_id = "Lance"

    # Search
    for memory in across_thread_memory.search(("todo", user_id)):
        print(memory.value)

    # We supply a thread ID for short-term (within-thread) memory
    # We supply a user ID for long-term (across-thread) memory
    config = {"configurable": {"thread_id": "2", "user_id": "Lance"}}

    # Chat with the chatbot
    input_messages = [
        HumanMessage(content="I have 30 minutes, what tasks can I get done?")
    ]

    # Run the graph
    for chunk in graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    # Chat with the chatbot
    input_messages = [
        HumanMessage(content="Yes, give me some options to call for swim lessons.")
    ]

    # Run the graph
    for chunk in graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()
