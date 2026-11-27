import os

import httpx2
import pytest
from mcp_server import mcp  # noqa: F401 (registers into the in-process client)

needs_groq = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
)


# --- tools ---
async def test_tool_listed(client):
    names = [t.name for t in await client.list_tools()]
    assert "add" in names
    assert "read_langchain_docs_page" in names


async def test_add(client):
    result = await client.call_tool("add", {"a": 2, "b": 3})
    assert result.data == 5


async def test_add_negative(client):
    result = await client.call_tool("add", {"a": -1, "b": 1})
    assert result.data == 0


async def test_read_page(client, network):
    result = await client.call_tool(
        "read_langchain_docs_page",
        {"path": "oss/python/langchain/short-term-memory"},
        raise_on_error=False,
    )
    assert not result.is_error
    assert "checkpointer" in result.data.lower()


async def test_read_page_strips_slashes(client, network):
    result = await client.call_tool(
        "read_langchain_docs_page",
        {"path": "/oss/python/langchain/short-term-memory/"},
        raise_on_error=False,
    )
    assert not result.is_error
    assert "checkpointer" in result.data.lower()


async def test_read_missing_page(client, network):
    result = await client.call_tool(
        "read_langchain_docs_page",
        {"path": "oss/python/this-page-does-not-exist"},
        raise_on_error=False,
    )
    assert result.is_error


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
    assert "agents" in result.messages[0].content.text


# --- LLM-backed tool ---
@needs_groq
async def test_ask_succeeds(client, network):
    result = await client.call_tool(
        "ask_langchain_docs",
        {"question": "What is LangGraph?"},
        raise_on_error=False,
    )
    assert not result.is_error
    assert isinstance(result.data, str)
    assert result.data.strip()


@needs_groq
async def test_ask_returns_answer_not_full_doc(client, network):
    doc = await client.read_resource("docs://langchain-learn")
    result = await client.call_tool(
        "ask_langchain_docs", {"question": "What is LangGraph?"}
    )
    assert len(result.data) < len(doc[0].text)


async def test_ask_missing_question(client):
    # Validation fails before any LLM call — no key or network needed
    result = await client.call_tool("ask_langchain_docs", {}, raise_on_error=False)
    assert result.is_error
