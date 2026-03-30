"""
Tests for AgentMirror ConsistencyJudge.
Run with: pytest tests/
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from agentmirror import ConsistencyJudge, ConsistencyReport, LightweightParaphraser, SemanticScorer
from langchain_core.messages import HumanMessage, AIMessage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_mock_graph(response_text: str = "The answer is 42."):
    """Returns a mock LangGraph graph that always responds with response_text."""
    graph = MagicMock()
    graph.invoke.return_value = {
        "messages": [AIMessage(content=response_text)]
    }
    return graph


def make_mock_llm(paraphrases: list[str] = None):
    """Returns a callable that returns a JSON array of paraphrases."""
    if paraphrases is None:
        paraphrases = [
            "Can you tell me: test query",
            "I need to know: test query",
            "Please explain: test query",
            "Help me understand: test query",
        ]
    return lambda _: json.dumps(paraphrases)


# ---------------------------------------------------------------------------
# LightweightParaphraser
# ---------------------------------------------------------------------------

class TestLightweightParaphraser:

    def test_returns_correct_number_of_variants(self):
        paraphrases = ["v1", "v2", "v3", "v4"]
        p = LightweightParaphraser(make_mock_llm(paraphrases), n_variants=4)
        result = p.generate("test query")
        assert len(result) == 4

    def test_falls_back_gracefully_on_bad_llm_response(self):
        bad_llm = lambda _: "this is not json"
        p = LightweightParaphraser(bad_llm, n_variants=4)
        result = p.generate("test query")
        assert len(result) == 4
        assert all(isinstance(v, str) for v in result)

    def test_respects_n_variants_limit(self):
        paraphrases = ["v1", "v2", "v3", "v4", "v5", "v6"]
        p = LightweightParaphraser(make_mock_llm(paraphrases), n_variants=3)
        result = p.generate("test query")
        assert len(result) == 3


# ---------------------------------------------------------------------------
# SemanticScorer
# ---------------------------------------------------------------------------

class TestSemanticScorer:

    def test_identical_texts_score_near_one(self):
        scorer = SemanticScorer()
        score = scorer.score("Hello world", "Hello world")
        assert score > 0.99

    def test_unrelated_texts_score_lower(self):
        scorer = SemanticScorer()
        score = scorer.score(
            "The capital of France is Paris.",
            "Quantum entanglement describes correlated particles."
        )
        assert score < 0.5

    def test_score_many_returns_correct_length(self):
        scorer = SemanticScorer()
        scores = scorer.score_many("reference text", ["text one", "text two", "text three"])
        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)


# ---------------------------------------------------------------------------
# ConsistencyJudge
# ---------------------------------------------------------------------------

class TestConsistencyJudge:

    def test_consistent_agent_scores_high(self):
        """An agent that gives the same answer to all variants should score near 1."""
        graph = make_mock_graph("Conservative investors should prioritize bonds and diversification.")
        judge = ConsistencyJudge(
            graph=graph,
            llm_callable=make_mock_llm(),
            n_variants=4,
            consistency_threshold=0.75,
        )
        report = judge.evaluate("What should a conservative investor do?")
        assert report.consistency_score > 0.90
        assert report.flagged is False

    def test_inconsistent_agent_gets_flagged(self):
        """An agent returning different answers to variants should be flagged."""
        responses = [
            "Invest in bonds for safety.",
            "The mitochondria is the powerhouse of the cell.",
            "Python is a great programming language.",
            "The weather in Bengaluru is usually warm.",
            "Consider diversifying your portfolio.",
        ]
        call_count = [0]

        def rotating_graph(state, config=None):
            i = call_count[0] % len(responses)
            call_count[0] += 1
            return {"messages": [AIMessage(content=responses[i])]}

        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = rotating_graph

        judge = ConsistencyJudge(
            graph=mock_graph,
            llm_callable=make_mock_llm(),
            n_variants=4,
            consistency_threshold=0.75,
        )
        report = judge.evaluate("What should a conservative investor do?")
        assert report.flagged is True
        assert report.flag_reason is not None

    def test_report_structure_is_complete(self):
        graph = make_mock_graph("A test response.")
        judge = ConsistencyJudge(
            graph=graph,
            llm_callable=make_mock_llm(),
            n_variants=3,
        )
        report = judge.evaluate("Test query")

        assert isinstance(report, ConsistencyReport)
        assert report.original_query == "Test query"
        assert report.num_variants == 3
        assert len(report.variants) == 3
        assert all(v.similarity_to_original is not None for v in report.variants)
        assert 0.0 <= report.consistency_score <= 1.0
        assert report.evaluated_at is not None
        assert report.evaluation_duration_ms > 0

    def test_report_json_export(self, tmp_path):
        graph = make_mock_graph("Test response")
        judge = ConsistencyJudge(
            graph=graph,
            llm_callable=make_mock_llm(),
            n_variants=2,
        )
        report = judge.evaluate("Test")
        path = str(tmp_path / "report.json")
        json_str = report.to_json(path)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["original_query"] == "Test"

        # Should be saved to file
        with open(path) as f:
            file_content = json.loads(f.read())
        assert file_content["original_query"] == "Test"

    def test_batch_evaluation_returns_one_report_per_query(self):
        graph = make_mock_graph("Response")
        judge = ConsistencyJudge(
            graph=graph,
            llm_callable=make_mock_llm(),
            n_variants=2,
        )
        queries = ["Query one", "Query two", "Query three"]
        reports = judge.evaluate_batch(queries)
        assert len(reports) == 3
        for r, q in zip(reports, queries):
            assert r.original_query == q
