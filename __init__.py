"""
AgentMirror
-----------
Production evaluation for multi-agent AI systems without ground truth.

Built for teams who have shipped real agents and need to know
if they're actually working well — not just running.

Modules:
    ConsistencyJudge  — evaluates agent reliability via behavioral consistency testing
    (ChainBlame, ReasoningScorer, JudgeCalibrator — coming soon)

Quickstart:
    from agentmirror import ConsistencyJudge

    judge = ConsistencyJudge(
        graph=your_compiled_langgraph_app,
        llm_callable=lambda p: your_llm.invoke(p).content,
    )
    report = judge.evaluate("What is the client's risk profile?")

    print(report.consistency_score)   # 0.0 - 1.0
    print(report.flagged)             # True if below threshold
    report.to_json("report.json")     # save for review
"""

from agentmirror.consistency_judge import (
    ConsistencyJudge,
    ConsistencyReport,
    LightweightParaphraser,
    SemanticScorer,
    VariantResult,
)

__all__ = [
    "ConsistencyJudge",
    "ConsistencyReport",
    "LightweightParaphraser",
    "SemanticScorer",
    "VariantResult",
]

__version__ = "0.1.0"
__author__ = "Naureen Fathima"
__description__ = "Production evaluation for multi-agent AI systems without ground truth."
