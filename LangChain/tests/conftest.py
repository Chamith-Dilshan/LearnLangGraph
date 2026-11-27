import socket
import sys
from pathlib import Path

import pytest
from fastmcp import Client

# Make both import roots available: LangChain/ (agent_with_mcp)
# and LangChain/mcp_server/ (mcp_server.py)
ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "mcp_server"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from mcp_server import mcp  # noqa: E402  (FastMCP instance)


def _network_available() -> bool:
    try:
        socket.create_connection(("docs.langchain.com", 443), timeout=3).close()
        return True
    except OSError:
        return False


@pytest.fixture
async def client():
    """In-process FastMCP client — no subprocess, no socket."""
    async with Client(mcp) as c:
        yield c


@pytest.fixture
def network():
    """Skip the test when docs.langchain.com is unreachable."""
    if not _network_available():
        pytest.skip("no network access")
