import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph

load_dotenv()

llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model="openai/gpt-oss-20b",
        temperature=0,
    )

class AgentState(MessagesState):
    summary: str

# Define the logic to call the model
def call_model(state: AgentState):
    # Get summary if it exists
    summary = state.get("summary", "")

    # If there is summary, then we add it
    if summary:

        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"

        # Append summary to any newer messages
        messages = [SystemMessage(content=system_message)] + state["messages"]

    else:
        messages = state["messages"]

    response = llm.invoke(messages)
    return {"messages": response}


def summarize_conversation(state: AgentState):
    # First, we get any existing summary
    summary = state.get("summary", "")

    # Create our summarization prompt
    if summary:

        # A summary already exists
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )

    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)

    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}


# Determine whether to end or summarize the conversation
def should_continue(state: AgentState) -> Literal["summarize_conversation", END]:
    """Return the next node to execute."""

    messages = state["messages"]

    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"

    # Otherwise we can just end
    return END

# Define a new graph
workflow = StateGraph(AgentState)
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

# Set the entrypoint as conversation
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

# Compile
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

if __name__ == '__main__':
    # Create a thread
    config = {"configurable": {"thread_id": "1"}}

    # Start conversation
    input_message = HumanMessage(content="hi! I'm Lance")
    output = graph.invoke({"messages": [input_message]}, config)
    for m in output['messages'][-1:]:
        m.pretty_print()

    input_message = HumanMessage(content="what's my name?")
    output = graph.invoke({"messages": [input_message]}, config)
    for m in output['messages'][-1:]:
        m.pretty_print()

    input_message = HumanMessage(content="i like the 49ers!")
    output = graph.invoke({"messages": [input_message]}, config)
    for m in output['messages'][-1:]:
        m.pretty_print()

    print("\nSummary: ", graph.get_state(config).values.get("summary",""))

    input_message = HumanMessage(content="i like Nick Bosa, do you like him?")
    output = graph.invoke({"messages": [input_message]}, config)
    for m in output['messages'][-1:]:
        m.pretty_print()

    print("\nSummary: ", graph.get_state(config).values.get("summary",""))