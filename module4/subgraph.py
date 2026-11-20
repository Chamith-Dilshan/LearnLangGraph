import os
from operator import add
from typing import Annotated, TypedDict

from langgraph.constants import END, START
from langgraph.graph import StateGraph

# The most critical thing to understand is how the graphs communicate
# In short, communication is done with overlapping keys:
# 1. The sub-graphs can access docs from the parent.
# 2. The parent can access summary/failure_report from the sub-graphs.


# The structure of the logs
class Log(TypedDict):
    id: str
    question: str
    docs: list | None
    answer: str
    grade: int | None
    grader: str | None
    feedback: str | None


# Sub Graphs
# Failure Analysis Sub-graph
class FailureAnalysisState(TypedDict):
    cleaned_logs: list[Log]
    failures: list[Log]
    fa_summary: str
    processed_logs: list[str]


class FailureAnalysisOutputState(TypedDict):
    fa_summary: str
    processed_logs: list[str]


def get_failures(state):
    """Get logs that contain a failure"""
    cleaned_logs = state["cleaned_logs"]
    failures = [log for log in cleaned_logs if "grade" in log]
    return {"failures": failures}


def generate_summary(state):
    """Generate a summary of failures"""
    failures = state["failures"]
    # Add fxn: fa_summary = summarize(failures)
    fa_summary = "Poor quality retrieval of Chroma documentation."
    return {
        "fa_summary": fa_summary,
        "processed_logs": [
            f"failure-analysis-on-log-{failure['id']}" for failure in failures
        ],
    }


fa_builder = StateGraph(
    state_schema=FailureAnalysisState, output_schema=FailureAnalysisOutputState
)
fa_builder.add_node("get_failures", get_failures)
fa_builder.add_node("generate_summary", generate_summary)

fa_builder.add_edge(START, "get_failures")
fa_builder.add_edge("get_failures", "generate_summary")
fa_builder.add_edge("generate_summary", END)

fa_graph = fa_builder.compile()


# Summarization subgraph
class QuestionSummarizationState(TypedDict):
    cleaned_logs: list[Log]
    qs_summary: str
    report: str
    processed_logs: list[str]


class QuestionSummarizationOutputState(TypedDict):
    report: str
    processed_logs: list[str]


def generate_summary(state):
    cleaned_logs = state["cleaned_logs"]
    # Add fxn: summary = summarize(generate_summary)
    summary = "Questions focused on usage of ChatOllama and Chroma vector store."
    return {
        "qs_summary": summary,
        "processed_logs": [f"summary-on-log-{log['id']}" for log in cleaned_logs],
    }


def send_to_slack(state):
    qs_summary = state["qs_summary"]
    # Add fxn: report = report_generation(qs_summary)
    report = "foo bar baz"
    return {"report": report}


qs_builder = StateGraph(
    QuestionSummarizationState, output_schema=QuestionSummarizationOutputState
)
qs_builder.add_node("generate_summary", generate_summary)
qs_builder.add_node("send_to_slack", send_to_slack)

qs_builder.add_edge(START, "generate_summary")
qs_builder.add_edge("generate_summary", "send_to_slack")
qs_builder.add_edge("send_to_slack", END)

qs_graph = qs_builder.compile()


# Main Graph
class MainGraphState(TypedDict):
    raw_logs: list[Log]
    cleaned_logs: list[Log]
    fa_summary: str  # This will only be generated in the FA sub-graph
    report: str  # This will only be generated in the QS sub-graph
    processed_logs: Annotated[
        list[int], add
    ]  # This will be generated in BOTH sub-graphs


def clean_logs(state):
    # Get logs
    raw_logs = state["raw_logs"]
    # Data cleaning raw_logs -> docs
    cleaned_logs = raw_logs
    return {"cleaned_logs": cleaned_logs}


main_builder = StateGraph(MainGraphState)
main_builder.add_node("clean_logs", clean_logs)
main_builder.add_node("question_summarization", qs_graph)
main_builder.add_node("failure_analysis", fa_graph)

main_builder.add_edge(START, "clean_logs")
main_builder.add_edge("clean_logs", "failure_analysis")
main_builder.add_edge("clean_logs", "question_summarization")
main_builder.add_edge("failure_analysis", END)
main_builder.add_edge("question_summarization", END)

main_graph = main_builder.compile()

if __name__ == "__main__":
    os.makedirs("images", exist_ok=True)
    file_path = os.path.join("images", "subgraph.png")

    if not os.path.exists(file_path):
        png_data = main_graph.get_graph().draw_mermaid_png()
        with open(file_path, "wb") as f:
            f.write(png_data)
        print(f"Image saved to {file_path}")

    # Dummy logs
    question_answer = Log(
        id="1",
        question="How can I import ChatOllama?",
        answer="To import ChatOllama, use: 'from langchain_community.chat_models import ChatOllama.'",
    )

    question_answer_feedback = Log(
        id="2",
        question="How can I use Chroma vector store?",
        answer="To use Chroma, define: rag_chain = create_retrieval_chain(retriever, question_answer_chain).",
        grade=0,
        grader="Document Relevance Recall",
        feedback="The retrieved documents discuss vector stores in general, but not Chroma specifically",
    )

    raw_logs = [question_answer, question_answer_feedback]
    result = main_graph.invoke({"raw_logs": raw_logs})
    print(f"Result: {result}")
