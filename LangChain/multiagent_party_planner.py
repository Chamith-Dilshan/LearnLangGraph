import asyncio
import json
import sqlite3

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain.mcp import MCPAdapter
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.types import Command
from pydantic import BaseModel, Field

load_dotenv()

tavily = TavilySearch(max_results=5)

DB_PATH = "playlist_db/playlist.db"


# ---------- Shared state and per-run context ----------


class PartyState(AgentState):
    origin: str
    destination: str
    guest_count: int
    genre: str


class PartyContext(BaseModel):
    """Details about the party planner user."""

    user_name: str = Field(default="user", description="The name of the user")
    home_city: str = Field(default="", description="The home city of the user")
    budget: str = Field(default="moderate", description="The budget of the user")


# ---------- Output schemas ----------


class FlightOption(BaseModel):
    airline: str
    price_usd: float
    duration_hours: float
    departure_date: str
    notes: str = ""


class TravelReport(BaseModel):
    """Shortlist of best flight options."""

    options: list[FlightOption]


class VenueOption(BaseModel):
    name: str
    location: str
    estimated_rate: str = Field(description="Estimated rate, e.g. '$1500/day'")
    capacity: int
    why_it_fits: str


class VenueReport(BaseModel):
    """Shortlist of the best venue options."""

    options: list[VenueOption]


class PlaylistEntry(BaseModel):
    position: int
    title: str
    artist: str
    genre: str
    bpm: int
    vibe: str
    reason: str = Field(description="Why this track fits the event sequence")


class MusicReport(BaseModel):
    """Music report with playlist entries."""

    genre: str
    entries: list[PlaylistEntry]


class PartyPlan(BaseModel):
    """Complete party plan returned by the coordinator."""

    origin: str
    destination: str
    guest_count: int
    genre: str
    flights: list[FlightOption]
    venues: list[VenueOption]
    playlist: list[PlaylistEntry]
    summary: str


# ---------- Database tools (schema-aware, per the SQL agent guide) ----------


@tool
def sql_db_list_tables() -> str:
    """Input is an empty string, output is a comma-separated list of tables in the playlist database."""
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()
        return ", ".join(r[0] for r in rows if not r[0].startswith("sqlite_"))


@tool
def sql_db_schema(table_names: str) -> str:
    """Input is a comma-separated list of tables, output is the schema and sample rows for those tables.
    Be sure that the tables actually exist by calling sql_db_list_tables first!"""
    results = []
    with sqlite3.connect(DB_PATH) as con:
        for name in table_names.split(","):
            name = name.strip()
            row = con.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (name,)
            ).fetchone()
            if row:
                results.append(row[0])
            else:
                results.append(f"Error: table {name!r} not found")
    return "\n\n".join(results)


@tool
def sql_db_query(query: str) -> str:
    """Input is a detailed and correct SQL query against the playlist database, output is the result.
    If an error is returned, rewrite the query and try again.
    Rank/filter on bpm, vibe, genre and energy to build playlists that match the event — never random."""
    try:
        with sqlite3.connect(DB_PATH) as con:
            rows = con.execute(query).fetchall()
            return str(rows)
    except Exception as e:
        return f"Error: {e}"


# ---------- Generic tools ----------


@tool
def web_search(query: str) -> dict:
    """Search the web for relevant information (venues, rates, locations)."""
    return tavily.invoke({"query": query})


# ---------- Prompts ----------

TRAVEL_AGENT_SYSTEM_PROMPT = """You are a travel agent. Search for flights to the destination.
Find the best one-way, one-ticket economy options by lowest price, shortest duration, and a date that
suits a party. You may search multiple times. If a tool fails or returns nothing usable, retry it.
Return a shortlist of the best options."""

MUSIC_AGENT_SYSTEM_PROMPT = """You are the music curator for a party planner.
First call sql_db_list_tables, then sql_db_schema on every table you need, THEN write SQL.
Use sql_db_query to select songs ordered by how well they match the requested genre, vibe, and
energy level, and use bpm to sequence the playlist: start low-bpm/chill, build to high-bpm/party
tracks. Always justify each pick with its bpm/vibe from the DB. Never invent songs that are not
in the query result."""

VENUE_AGENT_SYSTEM_PROMPT = """You are a venue scout. Use web_search to find candidate venues in the
destination for the given guest count, then compare them on estimated price/rate and fit against the
user's budget. Return a shortlist with rates and why each fits."""

COORDINATOR_SYSTEM_PROMPT = """You are the party planner coordinator. You have three specialist tools:
search_flights, search_venues, and suggest_playlist. Workflow:
1. Call update_party_details as soon as you know the origin, destination, guest count and genre.
2. Then call the specialist tools to gather flights, venues and a playlist. You may call several in one turn.
3. Combine everything into a clear plan. If the user's location is known, use it as the default origin."""


# ---------- Coordinator tool: state update (no subagent needed, module-level) ----------


# required if the party details can arrive mid-conversation instead of at run start.
@tool
def update_party_details(
    origin: str, destination: str, guest_count: int, genre: str, runtime: ToolRuntime
) -> Command:
    """Save the party details (origin city, destination city, guest count, music genre).
    Call this FIRST, before any other planning tool."""
    return Command(
        update={
            "origin": origin,
            "destination": destination,
            "guest_count": guest_count,
            "genre": genre,
            "messages": [
                ToolMessage("Party details saved.", tool_call_id=runtime.tool_call_id)
            ],
        }
    )


# ---------- Subagent builders (music and venue are static; travel needs MCP) ----------


def build_music_agent():
    return create_agent(
        model=ChatGroq(model="llama-3.1-8b-instant", temperature=0),
        tools=[sql_db_list_tables, sql_db_schema, sql_db_query],
        system_prompt=MUSIC_AGENT_SYSTEM_PROMPT,
        response_format=ToolStrategy(MusicReport),
    )


def build_venue_agent():
    return create_agent(
        model=ChatGroq(model="openai/gpt-oss-20b", temperature=0),
        tools=[web_search],
        system_prompt=VENUE_AGENT_SYSTEM_PROMPT,
        response_format=ToolStrategy(VenueReport),
    )


# ---------- Graph factory — Agent Server calls this once per run ----------


async def make_graph(config: RunnableConfig | None = None):
    """Graph factory — takes 0 or 1 injected argument (RunnableConfig)."""
    music_agent = build_music_agent()
    venue_agent = build_venue_agent()

    async with MCPAdapter("https://mcp.kiwi.com") as adapter:
        mcp_tools = await adapter.list_tools(cache_mode="use")
        travel_agent = create_agent(
            model=ChatGroq(model="llama-3.3-70b-versatile", temperature=0),
            tools=[*mcp_tools, web_search],
            system_prompt=TRAVEL_AGENT_SYSTEM_PROMPT,
            response_format=ToolStrategy(TravelReport),
        )

        # Wrapper tools close over the subagents — resolved at definition time,
        # no globals, and no extra factory arguments.

        @tool
        async def search_flights(runtime: ToolRuntime) -> str:
            """Find flight options from the party origin to the destination. Uses the Kiwi MCP travel tools."""
            result = await travel_agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content=f"Find one-way economy flights from "
                            f"{runtime.state['origin']} to {runtime.state['destination']}."
                        )
                    ]
                }
            )
            report = result.get("structured_response")
            if report is not None:
                return json.dumps(report.model_dump(), indent=2)
            return result["messages"][-1].content

        @tool
        def search_venues(runtime: ToolRuntime) -> str:
            """Search venues and their rates in the destination, matching the guest count and budget."""
            result = venue_agent.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content=f"Find venues in {runtime.state['destination']} "
                            f"for {runtime.state['guest_count']} guests"
                        )
                    ]
                }
            )
            report = result.get("structured_response")
            if report is not None:
                return json.dumps(report.model_dump(), indent=2)
            return result["messages"][-1].content

        @tool
        def suggest_playlist(runtime: ToolRuntime) -> str:
            """Query the playlist DB and build a bpm-ordered playlist matching the requested genre and vibe."""
            result = music_agent.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content=f"Build a bpm-ordered playlist for the party, "
                            f"primary genre: {runtime.state['genre']}"
                        )
                    ]
                }
            )
            report = result.get("structured_response")
            if report is not None:
                return json.dumps(report.model_dump(), indent=2)
            return result["messages"][-1].content

        # No checkpointer — Agent Server provisions persistence itself.
        # No user data in config — PartyContext travels via context= at invoke time.
        return create_agent(
            model=ChatGroq(model="openai/gpt-oss-120b", temperature=0),
            tools=[
                update_party_details,
                search_flights,
                search_venues,
                suggest_playlist,
            ],
            state_schema=PartyState,
            context_schema=PartyContext,
            system_prompt=COORDINATOR_SYSTEM_PROMPT,
            response_format=ToolStrategy(PartyPlan, handle_errors=True),
            middleware=[
                ModelCallLimitMiddleware(
                    run_limit=2,  # max 2 coordinator iterations per run
                    exit_behavior="end",  # stop gracefully when the cap is hit
                )
            ],
        )


# ---------- Local runner (same code path as Studio) ----------


async def main():
    coordinator = await make_graph()

    result = await coordinator.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Plan a wedding party for 60 guests in Santorini. I'm flying from Berlin. "
                        "Genre: deep house."
                    )
                )
            ]
        },
        config={"configurable": {"thread_id": "1"}},
        context=PartyContext(user_name="John", home_city="Berlin", budget="premium"),
    )

    plan: PartyPlan = result["structured_response"]
    print(plan.summary)
    for flight in plan.flights:
        print(f"{flight.airline}: ${flight.price_usd}, {flight.duration_hours}h")
    for venue in plan.venues:
        print(f"{venue.name}: {venue.estimated_rate} ({venue.capacity} cap)")
    for entry in plan.playlist:
        print(f"{entry.position}. {entry.title} — {entry.bpm} BPM ({entry.vibe})")


if __name__ == "__main__":
    asyncio.run(main())
