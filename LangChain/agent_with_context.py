from pprint import pprint

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

load_dotenv()

"""Runtime Context & Context Schema Guide

Overview:
    Runtime context provides read-only, per-invocation static metadata (such as 
    user IDs, request preferences, feature flags, or database connection handles) 
    passed into an agent run via the `context=` argument of `.invoke()`.

Key Properties:
    - Ephemeral: Lives only for the duration of a single `.invoke()` or `.stream()` call.
    - Read-Only: Tools and nodes read from `runtime.context` but do not use it to 
      persistently mutate or save records across multiple separate user sessions.

Real-World Use Cases:
    - Multi-Tenant User Isolation: Pass `user_id` and `organization_id` via context. 
      Tools read `runtime.context.user_id` to ensure data access remains strictly 
      scoped to the authorized user [Inside tools](https://docs.langchain.com/oss/python/langchain/runtime#inside-tools).
    - Per-Request Clients & Feature Flags: Inject active database transactions, 
      user-scoped API tokens, or feature flags (`enable_advanced_search: true`) 
      without hardcoding them globally.
    - Localization: Pass request metadata like `locale="ja-JP"` or `timezone="UTC"` 
      so tools format outputs appropriately for that exact execution [Static runtime context](https://docs.langchain.com/oss/python/concepts/context#static-runtime-context).

Why Use Context Instead of State?
    - Separation of Concerns: Chat state (`messages`) contains the conversation 
      transcript, is persistent via checkpointers, and is visible to the LLM. 
      Context is request-scoped, safe for sensitive dependencies, and keeps 
      transient metadata decoupled from chat history [Custom context](https://docs.langchain.com/oss/python/langchain/middleware/custom).
"""

llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)


class UserContext(BaseModel):
    """User context is a class that holds information about the user."""

    favourite_colour: str = Field(
        default="blue", description="The user's favourite colour"
    )
    least_favourite_colour: str = Field(
        default="yellow", description="The user's least favourite colour"
    )


@tool
def get_fav_colour(runtime: ToolRuntime[UserContext]) -> str:
    """Get the favourite colour of the user"""
    return runtime.context.favourite_colour


@tool
def get_least_fav_colour(runtime: ToolRuntime[UserContext]) -> str:
    """Get the least favourite colour of the user"""
    return runtime.context.least_favourite_colour


@tool
def update_favourite_colour(
    favourite_colour: str, runtime: ToolRuntime[UserContext]
) -> Command:
    """Update the favourite colour of the user in the state once they've revealed it."""
    return Command(
        update={
            "favourite_colour": favourite_colour,
            "messages": [
                ToolMessage(
                    "Successfully updated favourite colour",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


agent = create_agent(
    # model="groq:qwen/qwen3.8-27b",
    model=llm,
    checkpointer=InMemorySaver(),
    tools=[get_fav_colour, get_least_fav_colour, update_favourite_colour],
    context_schema=UserContext,
)

if __name__ == "__main__":
    response = agent.invoke(
        {"messages": [HumanMessage(content="My favourite colour is green")]},
        {"configurable": {"thread_id": "1"}},
        context=UserContext(),
    )

    pprint(response)
