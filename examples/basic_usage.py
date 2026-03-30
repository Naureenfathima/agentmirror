"""
AgentMirror — ConsistencyJudge Example
----------------------------------------
This example shows how to wrap a real LangGraph agent
and evaluate its consistency.

Run with:
    pip install agentmirror
    python examples/basic_usage.py
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from typing import TypedDict, Annotated
import operator

from agentmirror import ConsistencyJudge


# ---------------------------------------------------------------------------
# 1. Define a simple LangGraph agent (replace with your real graph)
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


llm = ChatAnthropic(model="claude-haiku-4-5-20251001")  # fast, cheap for eval


def financial_advisor_agent(state: AgentState) -> AgentState:
    """A simple single-node agent. Replace with your multi-agent graph."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# Build the graph
builder = StateGraph(AgentState)
builder.add_node("agent", financial_advisor_agent)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)
app = builder.compile()


# ---------------------------------------------------------------------------
# 2. Set up ConsistencyJudge
# ---------------------------------------------------------------------------

# llm_callable: any function that takes a string prompt and returns a string.
# This is used only for paraphrase generation — lightweight and fast.
llm_callable = lambda prompt: llm.invoke([HumanMessage(content=prompt)]).content

judge = ConsistencyJudge(
    graph=app,
    llm_callable=llm_callable,
    n_variants=4,                   # how many paraphrases to test
    consistency_threshold=0.75,     # flag anything below this
)


# ---------------------------------------------------------------------------
# 3. Evaluate a query
# ---------------------------------------------------------------------------

query = "What should a conservative investor with a 10-year horizon consider?"

print(f"\n{'='*60}")
print(f"Evaluating: {query}")
print(f"{'='*60}\n")

report = judge.evaluate(query)

print(f"Consistency Score : {report.consistency_score:.2f}  (threshold: {judge.threshold})")
print(f"Flagged           : {report.flagged}")
if report.flag_reason:
    print(f"Flag Reason       : {report.flag_reason}")
print(f"Variants Tested   : {report.num_variants}")
print(f"Eval Duration     : {report.evaluation_duration_ms}ms")
print(f"\nOriginal Response (truncated):")
print(f"  {report.original_response[:200]}...")

print(f"\nVariant Breakdown:")
for i, v in enumerate(report.variants, 1):
    print(f"  [{i}] sim={v.similarity_to_original:.2f} | {v.variant_query[:60]}...")

# ---------------------------------------------------------------------------
# 4. Save the report
# ---------------------------------------------------------------------------

report.to_json("consistency_report.json")
print(f"\nFull report saved to consistency_report.json")


# ---------------------------------------------------------------------------
# 5. Batch evaluation example
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("Batch Evaluation Example")
print(f"{'='*60}\n")

queries = [
    "What is dollar-cost averaging?",
    "How does portfolio rebalancing work?",
    "What is the difference between a Roth and traditional IRA?",
]

reports = judge.evaluate_batch(queries)

for r in reports:
    status = "🚨 FLAGGED" if r.flagged else "✅ CONSISTENT"
    print(f"  {status}  score={r.consistency_score:.2f}  | {r.original_query[:55]}...")
