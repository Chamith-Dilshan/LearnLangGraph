from agent_with_mcp import select_tools
from langchain.mcp import MCPAdapter
from mcp_server import mcp


class FakeTool:
    def __init__(self, name):
        self.name = name


FAKE_TOOLS = [
    FakeTool("local_add"),
    FakeTool("local_read_langchain_docs_page"),
    FakeTool("docs-langchain_search_docs_by_lang_chain"),
    FakeTool("reference-langchain_search_api"),
]


# --- tool selection (pure logic, no LLM, no network) ---
def test_docs_questions_exclude_reference_tools():
    tools = select_tools("how do I add short-term memory?", FAKE_TOOLS)
    assert all("reference" not in t.name for t in tools)


def test_api_questions_include_reference_tools():
    tools = select_tools("what are the parameters of create_agent?", FAKE_TOOLS)
    assert any("reference" in t.name for t in tools)


def test_local_tools_always_available():
    tools = select_tools("what is 2 + 3?", FAKE_TOOLS)
    assert any(t.name.startswith("local") for t in tools)


# --- adapter discovery (in-process, no network) ---
async def test_adapter_discovers_in_process_server():
    async with MCPAdapter(mcp) as adapter:
        names = [t.name for t in await adapter.list_tools()]
        assert any("read_langchain_docs_page" in n for n in names)
