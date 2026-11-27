import os
from operator import add
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Send
from pydantic import BaseModel

load_dotenv()
# Map-reduce operations are essential for efficient task decomposition and parallel processing.
#
# It has two phases:
# 1. Map - Break a task into smaller sub-tasks, processing each sub-task in parallel.
# 2. Reduce - Aggregate the results across all of the completed, parallelized sub-tasks.

# In this example we let the llm create a set of jokes based on the topic we give and then
# based on those topics create jokes. after that let it pick the best one. This process is
# kind a like divide and conquer.


# Prompts
subjects_prompt = """Generate a list of 3 sub-topics that are all related to this overall topic: {topic}."""
joke_prompt = """Generate a joke about {subject}"""
best_joke_prompt = """Below are a bunch of jokes about {topic}. Select the best one! Return the ID of the best one, starting 0 as the ID for the first joke. Jokes: \n\n  {jokes}"""

# llm
llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="qwen/qwen3.8-27b",
    temperature=0,
)


class Subjects(BaseModel):
    subjects: list[str]


class BestJoke(BaseModel):
    id: int


class OverallState(TypedDict):
    topic: str
    subjects: list
    jokes: Annotated[list, add]
    best_selected_joke: str


def generate_topics(state: OverallState):
    prompt = subjects_prompt.format(topic=state["topic"])
    response = llm.with_structured_output(Subjects, method="json_schema").invoke(prompt)
    return {"subjects": response.subjects}


# we use the Send to create a joke for each subject.
# This is very useful! It can automatically parallelize joke generation for any number of subjects.
def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]


# Joke generation (map)
class JokeState(TypedDict):
    subject: str


class Joke(BaseModel):
    joke: str


def generate_joke(state: JokeState):
    prompt = joke_prompt.format(subject=state["subject"])
    response = llm.with_structured_output(Joke, method="json_schema").invoke(prompt)
    return {"jokes": [response.joke]}


# Best joke selection (reduce)
def best_joke(state: OverallState):
    jokes = "\n\n".join(state["jokes"])
    prompt = best_joke_prompt.format(topic=state["topic"], jokes=jokes)
    response = llm.with_structured_output(BestJoke, method="json_schema").invoke(prompt)
    return {"best_selected_joke": state["jokes"][response.id]}


# Construct the graph: here we put everything together to construct our graph
builder = StateGraph(OverallState)

builder.add_node("generate_topics", generate_topics)
builder.add_node("generate_joke", generate_joke)
builder.add_node("best_joke", best_joke)

builder.add_edge(START, "generate_topics")
builder.add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])
builder.add_edge("generate_joke", "best_joke")
builder.add_edge("best_joke", END)

# Compile the graph
graph = builder.compile()


if __name__ == "__main__":
    os.makedirs("images", exist_ok=True)
    file_path = os.path.join("images", "map_reduce.png")

    if not os.path.exists(file_path):
        png_data = graph.get_graph().draw_mermaid_png()
        with open(file_path, "wb") as f:
            f.write(png_data)
        print(f"Image saved to {file_path}")

    # Call the graph: here we call it to generate a list of jokes
    for s in graph.stream({"topic": "animals"}):
        print(s)
