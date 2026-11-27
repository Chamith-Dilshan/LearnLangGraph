import asyncio

import httpx2
from dotenv import load_dotenv
from fastmcp import Context, FastMCP
from langchain_groq import ChatGroq

# import anthropic

load_dotenv()

llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)  # create once at module scope
# llm = anthropic.AsyncAnthropic()

mcp = FastMCP("Learn LangChain Server")

LEARN_URL = "https://docs.langchain.com/oss/python/learn.md"


@mcp.resource("docs://langchain-learn")
def langchain_learn() -> str:
    """Fetch the official "Learn" page for LangChain and LangGraph (Markdown)."""
    resp = httpx2.get(LEARN_URL)
    resp.raise_for_status()
    return resp.text


@mcp.tool
def add(a: int, b: int) -> int:
    """Add two integers and return their sum."""
    return a + b


@mcp.prompt
def summarize_doc(focus: str) -> str:
    """Generate a prompt asking the LLM to summarize the LangChain learning docs on a given focus."""
    return (
        "Summarize the official LangChain and LangGraph learning documentation, "
        f"focusing on {focus}."
    )


# @mcp.tool
# async def read_langchain_docs_page(path: str) -> str:
#     """Read one LangChain docs page as Markdown.
#
#     Example path: 'oss/python/langchain/short-term-memory'
#     """
#     clean = path.strip().strip("/")
#     url = f"https://docs.langchain.com/{clean}.md"
#     resp = await asyncio.to_thread(httpx2.get, url)
#     resp.raise_for_status()
#     return resp.text


@mcp.tool
async def ask_langchain_docs(question: str, ctx: Context) -> str:
    """Answer a question using the LangChain Learn docs."""
    result = await ctx.read_resource("docs://langchain-learn")
    doc = result.contents[0].content if result.contents else ""

    # response = await llm.messages.create(
    #     model="claude-sonnet-4-5",
    #     max_tokens=512,
    #     system="Answer only from the provided documentation.",
    #     messages=[{"role": "user", "content": f"{doc}\n\nQuestion: {question}"}],
    # )

    response = await llm.ainvoke(
        f"Answer only from this documentation:\n\n{doc}\n\nQuestion: {question}"
    )

    # return response.content[0].text
    return response.text


if __name__ == "__main__":
    mcp.run()
