# AgentMirror

**Production evaluation for multi-agent AI systems — without ground truth.**

Most LLM eval frameworks assume you have expected outputs to compare against.  
In production, you don't. Your agents answer thousands of real questions every day,  
and you have no oracle to tell you if the answers were actually good.

AgentMirror is built for that reality.

---

## The Problem

When you deploy a multi-agent system to production, you face four evaluation problems that standard tooling doesn't solve:

| Problem | What it looks like |
|---|---|
| **No ground truth** | Advisors ask questions. Agents respond. You can't tell if the answer was correct or just confident. |
| **Metric blindness** | You have logs, latency numbers, token counts. Nothing tells you if the *reasoning* was good. |
| **Multi-hop tracing** | A final answer is wrong. You can't tell which agent in the chain caused it. |
| **Evaluating the evaluator** | You use LLM-as-judge. But you don't trust the judge either. |

AgentMirror addresses each of these — starting with the hardest one.

---

## ConsistencyJudge

The first module. Core insight:

> **A reliable agent should give semantically equivalent answers to semantically equivalent questions.**

If paraphrasing the same query produces wildly different responses, the agent's reasoning is unstable — regardless of whether any individual answer looks correct.

This is called *behavioral consistency testing*. It requires no ground truth, no human labelers, and no expected outputs. It runs entirely on your existing agent.

### How it works

```
Input query
    ↓
Generate N paraphrased variants  (lightweight LLM call)
    ↓
Run all variants through your LangGraph agent
    ↓
Score semantic similarity across all responses  (sentence-transformers, local)
    ↓
Consistency score (0–1) + variance report
    ↓
Flag low-consistency responses for human review  ← the flag that doesn't disappear
```

---

## Quickstart

```bash
pip install agentmirror
```

```python
from agentmirror import ConsistencyJudge

judge = ConsistencyJudge(
    graph=your_compiled_langgraph_app,          # your existing graph, unchanged
    llm_callable=lambda p: llm.invoke(p).content,  # for paraphrase generation
    n_variants=4,
    consistency_threshold=0.75,
)

report = judge.evaluate("What is the client's current risk profile?")

print(report.consistency_score)   # 0.0 – 1.0
print(report.flagged)             # True if below threshold
print(report.flag_reason)         # human-readable explanation
report.to_json("report.json")     # save for async review
```

### Batch evaluation

```python
reports = judge.evaluate_batch([
    "What is dollar-cost averaging?",
    "How does portfolio rebalancing work?",
    "Explain the difference between Roth and traditional IRAs.",
])

for r in reports:
    status = "🚨 FLAGGED" if r.flagged else "✅ CONSISTENT"
    print(f"{status}  {r.consistency_score:.2f}  {r.original_query}")
```

---

## Report Structure

Every evaluation returns a `ConsistencyReport`:

```python
class ConsistencyReport(BaseModel):
    original_query: str
    original_response: str
    consistency_score: float        # mean semantic similarity across variants
    min_similarity: float           # worst-case variant
    max_similarity: float           # best-case variant
    flagged: bool                   # True if score < threshold
    flag_reason: Optional[str]      # human-readable explanation
    variants: list[VariantResult]   # full breakdown per paraphrase
    num_variants: int
    evaluated_at: str               # ISO timestamp
    evaluation_duration_ms: float
```

---

## Design Principles

**No ground truth required.** Works on any production agent without labeled data.

**Framework-native.** Wraps your LangGraph graph directly. Zero changes to your agent code.

**Local scoring.** Semantic similarity runs locally via `sentence-transformers`. No API calls for evaluation.

**Honest output.** Flags are actionable, not decorative. A flagged response includes the consistency score, the threshold, and the lowest-similarity variant — so you know exactly what to look at.

**Composable.** `ConsistencyJudge` is the first module. `ChainBlame`, `ReasoningScorer`, and `JudgeCalibrator` are on the roadmap — each addressing one more failure mode from the list above.

---

## Roadmap

- [x] `ConsistencyJudge` — behavioral consistency testing without ground truth
- [ ] `ChainBlame` — trace failures to the specific agent in a multi-hop chain
- [ ] `ReasoningScorer` — score reasoning quality, not just output text
- [ ] `JudgeCalibrator` — measure how much you can trust your LLM judge

---

## Why This Exists

This library was built from direct experience running a federated multi-agent orchestrator serving 10,000+ financial advisors at 250K+ queries per week.

When advisors flagged bad answers, those flags disappeared. There was no way to know which agent caused the failure, whether a prompt change made things better or worse, or how to evaluate quality at a scale where human review is impossible.

AgentMirror is the tool I needed and couldn't find.

---

## Contributing

Issues, PRs, and discussion are welcome.  
If you've hit any of the four problems above in production, this project is for you.

---

## License

MIT
