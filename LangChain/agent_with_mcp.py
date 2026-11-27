import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ToolCallRequest
from langchain.agents.structured_output import ToolStrategy
from langchain.mcp import MCPAdapter
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()

SERVER_PATH = (
    Path(__file__).parent / "mcp_server" / "mcp_server.py"
)  # adjust to the real layout

# Local server over stdio (subprocess), remote docs servers over HTTP.
CONFIG = {
    "mcpServers": {
        "local": {
            "command": sys.executable,  # same interpreter/venv as the agent
            "args": [str(SERVER_PATH)],
        },
        # "docs-langchain": {"url": "https://docs.langchain.com/mcp"},
        # "reference-langchain": {"url": "https://reference.langchain.com/mcp"},
    }
}

llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)

# SYSTEM_PROMPT = (
#     "You are a LangChain documentation assistant.\n"
#     "Retrieval policy — follow these steps in order:\n"
#     "1. Search the docs with the search tool.\n"
#     "2. Pick the most relevant result and READ the actual page with the "
#     "filesystem tool (query_docs_filesystem_docs_by_lang_chain). Never answer "
#     "from a search snippet or a page title alone.\n"
#     "3. If the first page doesn't fully answer the question, read one or two "
#     "more linked pages.\n"
#     "4. Only then produce your final answer.\n"
#     "Answering rules:\n"
#     "- Answer ONLY with information you retrieved from page contents you read. "
#     "Never use your own knowledge.\n"
#     "- Every claim must be backed by a source: the doc page path and the tool "
#     "you used.\n"
#     "- If the pages you read don't answer the question, say so in `answer`, "
#     "leave `sources` with what you did find, and set `complete` to false. "
#     "Do not guess."
# )

SYSTEM_PROMPT = (
    "You are a LangChain documentation assistant.\n"
    "Rules:\n"
    "1. Answer ONLY with information you retrieved from tool results. "
    "Never use your own knowledge.\n"
    "2. Every claim must be backed by a source: name the tool you used and "
    "the doc page or section it came from.\n"
    "3. If the tools returned nothing relevant, say so in `answer` and leave "
    "`sources` empty. Do not guess."
)


class StructuredAnswer(BaseModel):
    """Structured answer grounded in retrieved documentation."""

    answer: str = Field(description="The answer, based only on retrieved tool results")
    sources: list[str] = Field(
        description="Tool names and doc pages/sections backing the answer"
    )
    found: bool = Field(
        description="Whether relevant information was found in the docs"
    )
    complete: bool = Field(
        description="True only if the retrieved information fully answers the question"
    )


"""
how tool calling works at the protocol level: a tool is "a schema — name, description, and argument 
definitions — paired with a function" (Tool calling). The model can only choose among tools whose 
schemas are present in the request, so every model call in the agent loop includes the schema of 
every registered tool. The tool code and results are not sent — only schemas — but the docs MCP 
server's tools carry long descriptions, and with both servers registered that crossed your 
Groq token limit. The model does execute only the tools it chooses; the overhead is the 
schema listing, not the execution.

LangChain addresses exactly this with Dynamic tool selection:

1. Decorator-based (@wrap_model_call, @wrap_tool_call) — quick and simple for single-hook middleware
2. Class-based (AgentMiddleware with hook methods) — more powerful for complex middleware with multiple 
   hooks or configuration
"""


def _text_of(message) -> str:
    """Return the text content of a message, whether it's a string or a blocklist."""
    content = message.content
    if isinstance(content, str):
        return content
    # List of content blocks — keep only the text ones
    return " ".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def select_tools(query: str, mcp_tools: list) -> list:
    """Pick which MCP tools to expose for a query. Pure function, testable."""
    wants_reference = any(word in query for word in ("api", "signature", "parameter"))
    if wants_reference:
        return list(mcp_tools)
    return [t for t in mcp_tools if "reference" not in t.name]


class McpToolSelectionMiddleware(AgentMiddleware):
    """Injects only the MCP tools relevant to the current query (async)."""

    def __init__(self, mcp_tools: list):
        self.mcp_tools = mcp_tools

    async def awrap_model_call(self, request: ModelRequest, handler):
        # Async variant of wrap_model_call — required for ainvoke()/astream().
        # Route on the FIRST user message so selection is stable across the
        # run, even after tool results enter the message list.
        first_user = next(
            (m for m in request.state["messages"] if m.type == "human"), None
        )
        query = _text_of(first_user).lower() if first_user else ""
        selected = select_tools(query, self.mcp_tools)
        updated = request.override(tools=[*request.tools, *selected])
        return await handler(updated)

    async def awrap_tool_call(self, request: ToolCallRequest, handler):
        # Required for runtime-registered tools: route calls to the MCP tool.
        matching = next(
            (t for t in self.mcp_tools if t.name == request.tool_call["name"]), None
        )
        if matching is not None:
            return await handler(request.override(tool=matching))
        return await handler(request)


"""
When we use middleware, we can inject tools dynamically based on the user's query.
and we can also edit the cache options. The cache and its per-principal isolation are
configured on the client itself (Client(cache=...)); this only selects how discovery reads it. 
Defaults to "use" (the default): repeat runs read the cached tool
list within the server's TTL instead of re-listing from all servers.

Usage of ToolStrategy force model to follow the given response format instead of using "json_mode"
This necessary when you use Groq. Otherwise you will get an error.
"""


async def make_graph():
    """Graph factory — Agent Server calls this once per run."""
    async with MCPAdapter(CONFIG) as adapter:
        # cache_mode="use" (the default): repeat runs read the cached tool
        # list within the server's TTL instead of re-listing from all servers.
        mcp_tools = await adapter.list_tools(cache_mode="use")

        return create_agent(
            model=llm,
            tools=[],  # no static tools — the middleware injects MCP tools per call
            system_prompt=SYSTEM_PROMPT,
            response_format=ToolStrategy(
                StructuredAnswer
            ),  # force tool-calling, not JSON mode
            middleware=[McpToolSelectionMiddleware(mcp_tools)],
        )


if __name__ == "__main__":

    async def agent_run():
        config = {"configurable": {"thread_id": "1"}}
        agent = await make_graph()
        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Use the add tool to compute 2 + 3, and also tell me "
                            "how to add short-term memory to a LangChain agent."
                        ),
                    }
                ]
            },
            config=config,
        )
        print(result["structured_response"])

    asyncio.run(agent_run())
