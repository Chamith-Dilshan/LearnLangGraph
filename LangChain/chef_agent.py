from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

load_dotenv()
tavily_client = TavilySearch(max_results=5)


@tool
def web_search(query: str) -> dict[str, Any]:
    """Search the web for information related to the given query."""
    return tavily_client.invoke(query)


SYSTEM_PROMPT = """
You are a personal chef. The user will give you a list of ingredients they have left over in their house.
Using the web search tool, search the web for recipes that can be made with the ingredients they have.
Return recipe suggestions and eventually the recipe instructions to the user, if requested.
"""

model = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)

agent = create_agent(
    # model="groq:qwen/qwen3.8-27b",
    model=model,
    tools=[web_search],
    system_prompt=SYSTEM_PROMPT,
    # checkpointer=InMemorySaver(),
)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}

    response = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="I have some cheese, milk and some eggs. what can I make with them?"
                )
            ],
        },
        config=config,
    )

    print(response["messages"][-1].content)
