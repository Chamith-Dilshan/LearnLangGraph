import os

import httpx2
import pytest
from fastmcp import Client
from mcp_server import mcp

needs_groq = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
)


@pytest.fixture
async def client():
    async with Client(mcp) as c:
        yield c


# --- tool ---
async def test_add(client):
    result = await client.call_tool("add", {"a": 2, "b": 3})
    assert result.data == 5


async def test_add_negative(client):
    result = await client.call_tool("add", {"a": -1, "b": 1})
    assert result.data == 0


async def test_tool_listed(client):
    tools = await client.list_tools()
    assert any(t.name == "add" for t in tools)


# --- resource ---
async def test_resource_listed(client):
    resources = await client.list_resources()
    assert any(str(r.uri) == "docs://langchain-learn" for r in resources)


async def test_resource_read(client, monkeypatch):
    class FakeResp:
        text = "# Learn LangChain"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(httpx2, "get", lambda url: FakeResp())

    content = await client.read_resource("docs://langchain-learn")
    assert content[0].text == "# Learn LangChain"


# --- prompt ---
async def test_prompt_listed(client):
    prompts = await client.list_prompts()
    assert any(p.name == "summarize_doc" for p in prompts)


async def test_prompt_get(client):
    result = await client.get_prompt("summarize_doc", {"focus": "agents"})
    text = result.messages[0].content.text
    assert "agents" in text


@needs_groq
async def test_ask_succeeds(client):
    result = await client.call_tool(
        "ask_langchain_docs",
        {"question": "What is LangGraph?"},
        raise_on_error=False,
    )
    assert not result.is_error  # the "200" check
    assert isinstance(result.data, str)
    assert result.data.strip()


@needs_groq
async def test_ask_returns_answer_not_full_doc(client):
    doc = await client.read_resource("docs://langchain-learn")
    result = await client.call_tool(
        "ask_langchain_docs", {"question": "What is LangGraph?"}
    )
    assert len(result.data) < len(doc[0].text)


async def test_ask_missing_question(client):
    result = await client.call_tool("ask_langchain_docs", {}, raise_on_error=False)
    assert result.is_error
