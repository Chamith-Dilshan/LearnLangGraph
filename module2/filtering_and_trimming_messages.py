""""
Reducer

A practical challenge when working with messages is managing long-running conversations.
Long-running conversations result in high token usage and latency if we are not careful,
because we pass a growing list of messages to the model.
For this we can use RemoveMessage and the add_messages reducer.
"""
import os

from dotenv import load_dotenv
from langchain_core.messages import RemoveMessage, AIMessage, HumanMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_groq import ChatGroq
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph

load_dotenv()

llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model="openai/gpt-oss-20b",
        temperature=0,
    )

# Node
def filter_messages(state: MessagesState):
    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"messages": delete_messages}

def chat_model_node(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}

""" 
Filtering

As you can see you can make the llm get last message only of the messages 
In this case you don't need to filter messages node. This is called Filtering.
"""
# def chat_model_node(state: MessagesState):
#     return {"messages": [llm.invoke(state["messages"][-1:])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("filter", filter_messages)
builder.add_node("chat_model", chat_model_node)
builder.add_edge(START, "filter")
builder.add_edge("filter", "chat_model")
builder.add_edge("chat_model", END)
graph = builder.compile()

""" 
Trim messages

Another approach is to trim messages, based upon a set number of tokens.
This restricts the message history to a specified number of tokens.
"""
# Node
def chat_model_node2(state: MessagesState):
    messages = trim_messages(
            state["messages"],
            max_tokens=100,
            strategy="last",
            token_counter=count_tokens_approximately,
            allow_partial=False,
        )
    return {"messages": [llm.invoke(messages)]}

# Build graph
builder2 = StateGraph(MessagesState)
builder2.add_node("chat_model", chat_model_node2)
builder2.add_edge(START, "chat_model")
builder2.add_edge("chat_model", END)
graph2 = builder2.compile()

if __name__ == '__main__':
    # Message list with a preamble
    messages = [AIMessage("Hi.", name="Bot", id="1")]
    messages.append(HumanMessage("Hi.", name="Lance", id="2"))
    messages.append(AIMessage("So you said you were researching ocean mammals?", name="Bot", id="3"))
    messages.append(
        HumanMessage("Yes, I know about whales. But what others should I learn about?", name="Lance", id="4"))

    # Invoke
    # output = graph.invoke({'messages': messages})
    # for m in output['messages']:
    #     m.pretty_print()

    messages_out_trim = graph2.invoke({'messages': messages})
    for m in messages_out_trim['messages']:
        m.pretty_print()