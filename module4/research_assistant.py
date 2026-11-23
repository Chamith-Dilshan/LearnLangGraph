# let's build  a lightweight, multi-agent system around chat models that customizes the research process

"""
1. Source Selection
Users can choose any set of input sources for their research.

2. Planning
Users provide a topic, and the system generates a team of AI analysts, each focusing on one sub-topic.
Human-in-the-loop will be used to refine these sub-topics before research begins.

3. LLM Utilization
Each analyst will conduct in-depth interviews with an expert AI using the selected sources.
The interview will be a multi-turn conversation to extract detailed insights as shown in the STORM paper.
These interviews will be captured in a using sub-graphs with their internal state.

4. Research Process
Experts will gather information to answer analyst questions in parallel.
And all interviews will be conducted simultaneously through map-reduce.

5. Output Format
The gathered insights from each interview will be synthesized into a final report.
We'll use customizable prompts for the report, allowing for a flexible output format.

"""

import operator
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    get_buffer_string,
)
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="qwen/qwen3.8-27b",
    temperature=0,
)


# Generate Analysts: Human-In-The-Loop
class Analyst(BaseModel):
    affiliation: str = Field(description="The primary affiliations of the analyst.")
    name: str = Field(description="The name of the analyst.")
    role: str = Field(
        description="The role of the analyst in the context of the topic."
    )
    description: str = Field(
        description="Description of the analyst focus, concerns, and motives.",
    )

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"


class Perspectives(BaseModel):
    analysts: list[Analyst] = Field(
        description="Comprehensive list of analysts with their roles and affiliations.",
    )


class GenerateAnalystsState(TypedDict):
    topic: str  # Research topic
    max_analysts: int  # Number of analysts
    human_analyst_feedback: str  # Human feedback
    analysts: list[Analyst]  # Analyst asking questions


analyst_instructions = """You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

1. First, review the research topic:
{topic}

2. Examine any editorial feedback that has been optionally provided to guide creation of the analysts: 

{human_analyst_feedback}

3. Determine the most interesting themes based upon documents and / or feedback above.

4. Pick the top {max_analysts} themes.

5. Assign one analyst to each theme."""


def create_analysts(state: GenerateAnalystsState):
    """Create analysts"""

    topic = state["topic"]
    max_analysts = state["max_analysts"]
    human_analyst_feedback = state.get("human_analyst_feedback", "")

    # Enforce structured output
    structured_llm = llm.with_structured_output(Perspectives)

    # System message
    system_message = analyst_instructions.format(
        topic=topic,
        human_analyst_feedback=human_analyst_feedback,
        max_analysts=max_analysts,
    )

    # Generate question
    analysts = structured_llm.invoke(
        [SystemMessage(content=system_message)]
        + [HumanMessage(content="Generate the set of analysts.")]
    )

    # Write the list of analysis to state
    return {"analysts": analysts.analysts}


def human_feedback(state: GenerateAnalystsState):
    """No-op node that should be interrupted on"""


def should_continue(state: GenerateAnalystsState):
    """Return the next node to execute"""

    # Check if human feedback
    human_analyst_feedback = state.get("human_analyst_feedback", None)
    if human_analyst_feedback:
        return "create_analysts"

    # Otherwise end
    return END


# Add nodes and edges
builder = StateGraph(GenerateAnalystsState)

builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)

builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges(
    "human_feedback", should_continue, ["create_analysts", END]
)

# Compile
memory = MemorySaver()
graph = builder.compile(interrupt_before=["human_feedback"], checkpointer=memory)


# Generate Question
# The analyst will ask questions to the expert.
class InterviewState(MessagesState):
    max_num_turns: int  # Number turns of conversation
    context: Annotated[list, operator.add]  # Source docs
    analyst: Analyst  # Analyst asking questions
    interview: str  # Interview transcript
    sections: list  # Final key we duplicate in outer state for Send() API


class SearchQuery(BaseModel):
    search_query: str | None = Field(None, description="Search query for retrieval.")


question_instructions = """You are an analyst tasked with interviewing an expert to learn about a specific topic. 

Your goal is boil down to interesting and specific insights related to your topic.

1. Interesting: Insights that people will find surprising or non-obvious.

2. Specific: Insights that avoid generalities and include specific examples from the expert.

Here is your topic of focus and set of goals: {goals}

Begin by introducing yourself using a name that fits your persona, and then ask your question.

Continue to ask questions to drill down and refine your understanding of the topic.

When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""


def generate_question(state: InterviewState):
    """Node to generate a question"""

    # Get state
    analyst = state["analyst"]
    messages = state["messages"]

    # Generate question
    system_message = question_instructions.format(goals=analyst.persona)
    question = llm.invoke([SystemMessage(content=system_message)] + messages)

    # Write messages to state
    return {"messages": [question]}


"""
Generate Answer: Parallelization
The expert will gather information from multiple sources in parallel to answer questions.
For example, we can use:

Specific web sites
Indexed documents e.g., via RAG
Web search
Wikipedia search
"""

# Search query writing
search_instructions = SystemMessage(
    content="""You will be given a conversation between an analyst and an expert. 

Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.

First, analyze the full conversation.

Pay particular attention to the final question posed by the analyst.

Convert this final question into a well-structured web search query"""
)

tavily_search = TavilySearch(max_results=3)


def search_web(state: InterviewState):
    """Retrieve docs from web search"""

    # Search query
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions] + state["messages"])

    # Search
    # search_docs = tavily_search.invoke(search_query.search_query) # updated 1.0
    writer = get_stream_writer()  # emit the real API query to the console log
    writer({"node": "search_web", "query": search_query.search_query})

    data = tavily_search.invoke({"query": search_query.search_query})
    search_docs = data.get("results", data)

    # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}


def search_wikipedia(state: InterviewState):
    """Retrieve docs from wikipedia"""

    # Search query
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions] + state["messages"])

    # Search
    writer = get_stream_writer()  # same here
    writer({"node": "search_wikipedia", "query": search_query.search_query})

    search_docs = WikipediaLoader(
        query=search_query.search_query, load_max_docs=2
    ).load()

    # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}


answer_instructions = """You are an expert being interviewed by an analyst.

Here is analyst area of focus: {goals}. 

You goal is to answer a question posed by the interviewer.

To answer question, use this context:

{context}

When answering questions, follow these guidelines:

1. Use only the information provided in the context. 

2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.

3. The context contain sources at the topic of each individual document.

4. Include these sources your answer next to any relevant statements. For example, for source # 1 use [1]. 

5. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc

6. If the source is: <Document source="assistant/docs/llama3_1.pdf" page="7"/>' then just list: 

[1] assistant/docs/llama3_1.pdf, page 7 

And skip the addition of the brackets as well as the Document source preamble in your citation."""


def generate_answer(state: InterviewState):
    """Node to answer a question"""

    # Get state
    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]

    # Answer question
    system_message = answer_instructions.format(goals=analyst.persona, context=context)
    answer = llm.invoke([SystemMessage(content=system_message)] + messages)

    # Name the message as coming from the expert
    answer.name = "expert"

    # Append it to state
    return {"messages": [answer]}


def save_interview(state: InterviewState):
    """Save interviews"""

    # Get messages
    messages = state["messages"]

    # Convert interview to a string
    interview = get_buffer_string(messages)

    # Save to interviews key
    return {"interview": interview}


def route_messages(state: InterviewState, name: str = "expert"):
    """Route between question and answer"""

    # Get messages
    messages = state["messages"]
    max_num_turns = state.get("max_num_turns", 2)

    # Check the number of expert answers
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    # End if expert has answered more than the max turns
    if num_responses >= max_num_turns:
        return "save_interview"

    # This router is run after each question - answer pair
    # Get the last question asked to check if it signals the end of discussion
    last_question = messages[-2]

    if "Thank you so much for your help" in last_question.content:
        return "save_interview"
    return "ask_question"


section_writer_instructions = """You are an expert technical writer. 

Your task is to create a short, easily digestible section of a report based on a set of source documents.

1. Analyze the content of the source documents: 
- The name of each source document is at the start of the document, with the <Document tag.

2. Create a report structure using markdown formatting:
- Use ## for the section title
- Use ### for sub-section headers

3. Write the report following this structure:
a. Title (## header)
b. Summary (### header)
c. Sources (### header)

4. Make your title engaging based upon the focus area of the analyst: 
{focus}

5. For the summary section:
- Set up summary with general background / context related to the focus area of the analyst
- Emphasize what is novel, interesting, or surprising about insights gathered from the interview
- Create a numbered list of source documents, as you use them
- Do not mention the names of interviewers or experts
- Aim for approximately 400 words maximum
- Use numbered sources in your report (e.g., [1], [2]) based on information from source documents

6. In the Sources section:
- Include all sources used in your report
- Provide full links to relevant websites or specific document paths
- Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown.
- It will look like:

### Sources
[1] Link or Document name
[2] Link or Document name

7. Be sure to combine sources. For example this is not correct:

[3] https://ai.meta.com/blog/meta-llama-3-1/
[4] https://ai.meta.com/blog/meta-llama-3-1/

There should be no redundant sources. It should simply be:

[3] https://ai.meta.com/blog/meta-llama-3-1/

8. Final review:
- Ensure the report follows the required structure
- Include no preamble before the title of the report
- Check that all guidelines have been followed"""


def write_section(state: InterviewState):
    """Node to answer a question"""

    # Get state
    interview = state["interview"]
    context = state["context"]
    analyst = state["analyst"]

    # Write section using either the gathered source docs from interview (context) or the interview itself (interview)
    system_message = section_writer_instructions.format(focus=analyst.description)
    section = llm.invoke(
        [SystemMessage(content=system_message)]
        + [HumanMessage(content=f"Use this source to write your section: {context}")]
    )

    # Append it to state
    return {"sections": [section.content]}


# Add nodes and edges
interview_builder = StateGraph(InterviewState)
interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node("search_web", search_web)
interview_builder.add_node("search_wikipedia", search_wikipedia)
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_section", write_section)

# Flow
interview_builder.add_edge(START, "ask_question")
interview_builder.add_edge("ask_question", "search_web")
interview_builder.add_edge("ask_question", "search_wikipedia")
interview_builder.add_edge("search_web", "answer_question")
interview_builder.add_edge("search_wikipedia", "answer_question")
interview_builder.add_conditional_edges(
    "answer_question", route_messages, ["ask_question", "save_interview"]
)
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)

# Interview
interview_memory = MemorySaver()
interview_graph = interview_builder.compile(checkpointer=interview_memory).with_config(
    run_name="Conduct Interviews"
)

"""
Parallelization the interviews using Map-Reduce

1. We parallelize the interviews via the Send() API, a map step.
2. We combine them into the report body in a reduce step.
"""


class ResearchGraphState(TypedDict):
    topic: str  # Research topic
    max_analysts: int  # Number of analysts
    human_analyst_feedback: str  # Human feedback
    analysts: list[Analyst]  # Analyst asking questions
    sections: Annotated[list, operator.add]  # Send() API key
    introduction: str  # Introduction for the final report
    content: str  # Content for the final report
    conclusion: str  # Conclusion for the final report
    final_report: str  # Final report


def initiate_all_interviews(state: ResearchGraphState):
    """This is the "map" step where we run each interview sub-graph using Send API"""

    # Check if human feedback
    human_analyst_feedback = state.get("human_analyst_feedback")
    if human_analyst_feedback:
        # Return to create_analysts
        return "create_analysts"

    # Otherwise kick off interviews in parallel via Send() API
    else:
        topic = state["topic"]
        return [
            Send(
                "conduct_interview",
                {
                    "analyst": analyst,
                    "messages": [
                        HumanMessage(
                            content=f"So you said you were writing an article on {topic}?"
                        )
                    ],
                },
            )
            for analyst in state["analysts"]
        ]


report_writer_instructions = """You are a technical writer creating a report on this overall topic: 

{topic}

You have a team of analysts. Each analyst has done two things: 

1. They conducted an interview with an expert on a specific sub-topic.
2. They write up their finding into a memo.

Your task: 

1. You will be given a collection of memos from your analysts.
2. Think carefully about the insights from each memo.
3. Consolidate these into a crisp overall summary that ties together the central ideas from all of the memos. 
4. Summarize the central points in each memo into a cohesive single narrative.

To format your report:

1. Use markdown formatting. 
2. Include no pre-amble for the report.
3. Use no sub-heading. 
4. Start your report with a single title header: ## Insights
5. Do not mention any analyst names in your report.
6. Preserve any citations in the memos, which will be annotated in brackets, for example [1] or [2].
7. Create a final, consolidated list of sources and add to a Sources section with the `## Sources` header.
8. List your sources in order and do not repeat.

[1] Source 1
[2] Source 2

Here are the memos from your analysts to build your report from: 

{context}"""


def write_report(state: ResearchGraphState):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report
    system_message = report_writer_instructions.format(
        topic=topic, context=formatted_str_sections
    )
    report = llm.invoke(
        [SystemMessage(content=system_message)]
        + [HumanMessage(content=f"Write a report based upon these memos.")]
    )
    return {"content": report.content}


intro_conclusion_instructions = """You are a technical writer finishing a report on {topic}

You will be given all of the sections of the report.

You job is to write a crisp and compelling introduction or conclusion section.

The user will instruct you whether to write the introduction or conclusion.

Include no pre-amble for either section.

Target around 100 words, crisply previewing (for introduction) or recapping (for conclusion) all of the sections of the report.

Use markdown formatting. 

For your introduction, create a compelling title and use the # header for the title.

For your introduction, use ## Introduction as the section header. 

For your conclusion, use ## Conclusion as the section header.

Here are the sections to reflect on for writing: {formatted_str_sections}"""


def write_introduction(state: ResearchGraphState):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report

    instructions = intro_conclusion_instructions.format(
        topic=topic, formatted_str_sections=formatted_str_sections
    )
    intro = llm.invoke(
        [instructions] + [HumanMessage(content=f"Write the report introduction")]
    )
    return {"introduction": intro.content}


def write_conclusion(state: ResearchGraphState):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report

    instructions = intro_conclusion_instructions.format(
        topic=topic, formatted_str_sections=formatted_str_sections
    )
    conclusion = llm.invoke(
        [instructions] + [HumanMessage(content=f"Write the report conclusion")]
    )
    return {"conclusion": conclusion.content}


def finalize_report(state: ResearchGraphState):
    """Theis is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion"""
    # Save full final report
    content = state["content"]
    if content.startswith("## Insights"):
        content = content.strip("## Insights")
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None
    else:
        sources = None

    final_report = (
        state["introduction"]
        + "\n\n---\n\n"
        + content
        + "\n\n---\n\n"
        + state["conclusion"]
    )
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources
    return {"final_report": final_report}


# Add nodes and edges
main_builder = StateGraph(ResearchGraphState)
main_builder.add_node("create_analysts", create_analysts)
main_builder.add_node("human_feedback", human_feedback)
main_builder.add_node("conduct_interview", interview_graph)
main_builder.add_node("write_report", write_report)
main_builder.add_node("write_introduction", write_introduction)
main_builder.add_node("write_conclusion", write_conclusion)
main_builder.add_node("finalize_report", finalize_report)

# Logic
main_builder.add_edge(START, "create_analysts")
main_builder.add_edge("create_analysts", "human_feedback")
main_builder.add_conditional_edges(
    "human_feedback", initiate_all_interviews, ["create_analysts", "conduct_interview"]
)
main_builder.add_edge("conduct_interview", "write_report")
main_builder.add_edge("conduct_interview", "write_introduction")
main_builder.add_edge("conduct_interview", "write_conclusion")
main_builder.add_edge(
    ["write_conclusion", "write_report", "write_introduction"], "finalize_report"
)
main_builder.add_edge("finalize_report", END)

# Compile
main_memory = MemorySaver()
main_graph = main_builder.compile(
    interrupt_before=["human_feedback"], checkpointer=main_memory
)

if __name__ == "__main__":
    os.makedirs("images", exist_ok=True)
    file_path1 = os.path.join("images", "analysts_create_assistant.png")
    file_path2 = os.path.join("images", "interview_assistant.png")
    file_path3 = os.path.join("images", "research_assistant.png")

    if not os.path.exists(file_path1):
        png_data = graph.get_graph().draw_mermaid_png()
        with open(file_path1, "wb") as f:
            f.write(png_data)
        print(f"Image saved to {file_path1}")

    if not os.path.exists(file_path2):
        png_data = interview_graph.get_graph().draw_mermaid_png()
        with open(file_path2, "wb") as f:
            f.write(png_data)
        print(f"Image saved to {file_path2}")

    if not os.path.exists(file_path3):
        png_data = main_graph.get_graph().draw_mermaid_png()
        with open(file_path3, "wb") as f:
            f.write(png_data)
        print(f"Image saved to {file_path3}")

    thread = {"configurable": {"thread_id": "1"}}

    def banner(title: str) -> None:
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)

    def print_analyst(analyst) -> None:
        print(f"    Name        : {analyst.name}")
        print(f"    Affiliation : {analyst.affiliation}")
        print(f"    Role        : {analyst.role}")
        print(f"    Description : {analyst.description}")
        print("    " + "-" * 50)

    def log_stream(inputs: dict | None, label: str) -> None:
        """Stream node updates, logging each node step, until the graph pauses or ends."""
        banner(label)
        for chunk in graph.stream(inputs, thread, stream_mode="updates"):
            for node_name, update in chunk.items():
                # Pause signal. Do NOT index into `update` — with static
                # interrupt_before breakpoints it can be an empty tuple.
                if node_name == "__interrupt__":
                    print("\n  [INTERRUPT] Graph paused")
                    continue
                if not isinstance(update, dict):
                    # Defensive: log anything unexpected instead of crashing
                    print(f"\n  [node executed] {node_name}\n    {update!r}")
                    continue
                print(f"\n  [node executed] {node_name}")
                if update.get("analysts"):
                    print(f"    -> generated {len(update['analysts'])} analysts:")
                    for analyst in update["analysts"]:
                        print_analyst(analyst)
                for key, value in update.items():
                    if key != "analysts":
                        print(f"    {key}: {value!r}")

        # get_state is the authoritative source for where the graph stopped
        state = graph.get_state(thread)
        if state.next:
            print(f"\n  [PAUSED] Next node: {state.next} (waiting for human feedback)")
        else:
            print("\n  [DONE] Graph reached END")

    # ---- Phase 1: initial run -> creates analysts, interrupts before human_feedback
    log_stream(
        {
            "topic": "The benefits of adopting LangGraph as an agent framework",
            "max_analysts": 3,
        },
        "PHASE 1: Initial analyst generation",
    )

    # ---- Phase 2: human gives feedback -> analysts are regenerated, interrupt again
    feedback = "Add in someone from a startup to add an entrepreneur perspective"
    graph.update_state(
        thread, {"human_analyst_feedback": feedback}, as_node="human_feedback"
    )
    log_stream(None, f"PHASE 2: Resumed with feedback: '{feedback}'")

    # ---- Phase 3: human approves (no feedback) -> should_continue routes to END
    graph.update_state(
        thread, {"human_analyst_feedback": None}, as_node="human_feedback"
    )
    log_stream(None, "PHASE 3: Resumed with no feedback (approve)")

    final_state = graph.get_state(thread)
    print(f"\nFinal next: {final_state.next}")  # should print () i.e., empty

    # Conduct Interview - The analyst will ask the expert questions.

    def preview(text: str, width: int = 110) -> str:
        clean = " ".join(text.split())
        return clean[:width] + ("..." if len(clean) > width else "")

    def log_interview_stream(inputs: dict | None, config: dict, label: str) -> None:
        """Log the interview graph step by step: conversation turns + API search calls."""
        banner(label)
        for mode, chunk in interview_graph.stream(
            inputs, config, stream_mode=["updates", "custom"]
        ):
            if mode == "custom":
                print(f"    [api call] {chunk['node']}: '{chunk['query']}'")
                continue
            for node_name, update in chunk.items():
                if node_name == "__interrupt__":
                    print("\n  [INTERRUPT] Graph paused")
                    continue
                if not isinstance(update, dict):
                    print(f"\n  [node] {node_name}: {update!r}")
                    continue
                print(f"\n  [node] {node_name}")
                if update.get("messages"):
                    for msg in update["messages"]:
                        # messages are named "expert" by generate_answer;
                        # anything else is the analyst speaking
                        role = getattr(msg, "name", None) or "analyst"
                        print(f"    {role}: {preview(msg.content)}")
                if update.get("context"):
                    print(f"    -> gathered {len(update['context'])} source doc(s)")
                if update.get("interview"):
                    print(f"    -> transcript saved ({len(update['interview'])} chars)")
                if update.get("sections"):
                    print(
                        f"    -> section written ({len(update['sections'][0])} chars)"
                    )

    analyst = Analyst(
        affiliation="Acme Research",
        name="Lily Andrews",
        role="Machine Learning Engineer",
        description="Interested in multi-agent frameworks and human-in-the-loop patterns for LLM applications.",
    )

    interview_config = {"configurable": {"thread_id": "interview-1"}}
    log_interview_stream(
        {
            "analyst": analyst,
            "messages": [
                HumanMessage(
                    content="So you said you were writing an article on LangGraph?"
                )
            ],
            "max_num_turns": 2,
        },
        interview_config,
        "INTERVIEW 1: Lily Andrews",
    )

    final = interview_graph.get_state(interview_config).values
    print("\n" + "=" * 70)
    print("  FINAL SECTION")
    print("=" * 70)
    for section in final.get("sections", []):
        print(section)

    # Research Assistance Multi-agent system
    # Inputs
    max_analysts = 3
    topic = "The benefits of adopting LangGraph as an agent framework"
    main_thread = {"configurable": {"thread_id": "1"}}

    # Run the graph until the first interruption
    for event in main_graph.stream(
        {"topic": topic, "max_analysts": max_analysts},
        main_thread,
        stream_mode="updates",
    ):
        analysts = event.get("analysts", "")
        if analysts:
            for analyst in analysts:
                print(f"Name: {analyst.name}")
                print(f"Affiliation: {analyst.affiliation}")
                print(f"Role: {analyst.role}")
                print(f"Description: {analyst.description}")
                print("-" * 50)

                # We now update the state as if we are the human_feedback node
    main_graph.update_state(
        main_thread,
        {"human_analyst_feedback": "Add in the CEO of gen ai native startup"},
        as_node="human_feedback",
    )

    # Check
    for event in main_graph.stream(None, main_thread, stream_mode="updates"):
        analysts = event.get("analysts", "")
        if analysts:
            for analyst in analysts:
                print(f"Name: {analyst.name}")
                print(f"Affiliation: {analyst.affiliation}")
                print(f"Role: {analyst.role}")
                print(f"Description: {analyst.description}")
                print("-" * 50)

    # Confirm we are happy
    main_graph.update_state(
        main_thread, {"human_analyst_feedback": None}, as_node="human_feedback"
    )

    # Continue
    for event in main_graph.stream(None, main_thread, stream_mode="updates"):
        print("--Node--")
        node_name = next(iter(event.keys()))
        print(node_name)
