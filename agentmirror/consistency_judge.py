"""
AgentMirror - ConsistencyJudge
-------------------------------
Wraps a LangGraph graph and evaluates agent consistency
by testing semantic equivalence across paraphrased inputs.

Core insight: a reliable agent should give semantically equivalent
answers to semantically equivalent questions. If it doesn't, the
reasoning is unstable — not just variable.

Author: Built for production multi-agent systems without ground truth.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Callable, Optional

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util


# ---------------------------------------------------------------------------
# Output Models
# ---------------------------------------------------------------------------

class VariantResult(BaseModel):
    """Result for a single paraphrased variant of the original query."""
    variant_query: str
    agent_response: str
    response_time_ms: float
    similarity_to_original: Optional[float] = None  # set after scoring


class ConsistencyReport(BaseModel):
    """Full consistency evaluation report for one query."""
    original_query: str
    original_response: str
    consistency_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Mean semantic similarity across all variant responses. 1.0 = perfectly consistent."
    )
    min_similarity: float = Field(..., description="Worst-case variant similarity.")
    max_similarity: float = Field(..., description="Best-case variant similarity.")
    flagged: bool = Field(..., description="True if consistency_score < threshold.")
    flag_reason: Optional[str] = None
    variants: list[VariantResult]
    num_variants: int
    evaluated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    evaluation_duration_ms: float

    def to_json(self, path: Optional[str] = None) -> str:
        """Export report to JSON string, optionally saving to file."""
        output = self.model_dump_json(indent=2)
        if path:
            with open(path, "w") as f:
                f.write(output)
        return output


# ---------------------------------------------------------------------------
# Paraphrase Generator  (lightweight — uses the same LLM your graph uses)
# ---------------------------------------------------------------------------

class LightweightParaphraser:
    """
    Generates paraphrases using a small, fast LLM call.
    Designed to be model-agnostic — pass any callable that takes
    a string prompt and returns a string response.
    """

    SYSTEM_PROMPT = """You are a paraphrase generator. Given a query, produce {n} 
semantically equivalent paraphrases that preserve the full meaning but vary 
the wording, structure, and phrasing. Return ONLY a JSON array of strings. 
No explanation, no markdown, no extra text."""

    def __init__(self, llm_callable: Callable[[str], str], n_variants: int = 4):
        """
        Args:
            llm_callable: A function that takes a prompt string and returns
                          a string response. Works with any LLM.
                          Example: lambda p: llm.invoke(p).content
            n_variants:   How many paraphrases to generate per query.
        """
        self.llm_callable = llm_callable
        self.n_variants = n_variants

    def generate(self, query: str) -> list[str]:
        """Return a list of paraphrased variants of the query."""
        prompt = (
            self.SYSTEM_PROMPT.format(n=self.n_variants)
            + f"\n\nOriginal query: {query}"
        )
        try:
            raw = self.llm_callable(prompt)
            # Strip markdown fences if the model adds them
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            variants = json.loads(raw)
            if not isinstance(variants, list):
                raise ValueError("Expected a JSON array.")
            return [str(v) for v in variants[: self.n_variants]]
        except Exception as e:
            # Graceful degradation: return simple lexical variants
            print(f"[AgentMirror] Paraphrase generation failed ({e}). Using fallback variants.")
            return self._fallback_variants(query)

    def _fallback_variants(self, query: str) -> list[str]:
        """Simple rule-based fallback if LLM call fails."""
        return [
            f"Can you tell me: {query}",
            f"I need to know: {query}",
            f"Please explain: {query}",
            f"Help me understand: {query}",
        ][: self.n_variants]


# ---------------------------------------------------------------------------
# Semantic Similarity Scorer
# ---------------------------------------------------------------------------

class SemanticScorer:
    """
    Scores semantic similarity between text pairs using
    sentence-transformers all-MiniLM-L6-v2.
    Fast, local, no API cost.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)

    def score(self, text_a: str, text_b: str) -> float:
        """Returns cosine similarity in [0, 1]."""
        embeddings = self._model.encode([text_a, text_b], convert_to_tensor=True)
        similarity = util.cos_sim(embeddings[0], embeddings[1])
        return round(float(similarity.item()), 4)

    def score_many(self, reference: str, candidates: list[str]) -> list[float]:
        """Score a list of candidates against a single reference."""
        return [self.score(reference, c) for c in candidates]


# ---------------------------------------------------------------------------
# ConsistencyJudge  — the main interface
# ---------------------------------------------------------------------------

class ConsistencyJudge:
    """
    Wraps a LangGraph graph and evaluates whether it produces
    consistent responses to semantically equivalent queries.

    Usage:
        judge = ConsistencyJudge(
            graph=your_langgraph_app,
            llm_callable=lambda p: your_llm.invoke(p).content,
        )
        report = judge.evaluate("What is the client's current asset allocation?")
        print(report.consistency_score)
        print(report.flagged)
        report.to_json("report.json")
    """

    def __init__(
        self,
        graph: Any,
        llm_callable: Callable[[str], str],
        n_variants: int = 4,
        consistency_threshold: float = 0.75,
        input_key: str = "messages",
        output_key: str = "messages",
        graph_config: Optional[dict] = None,
    ):
        """
        Args:
            graph:                  Your compiled LangGraph graph (app).
            llm_callable:           Callable for paraphrase generation.
                                    Signature: (prompt: str) -> str
            n_variants:             Number of paraphrases to test.
            consistency_threshold:  Score below this flags the response.
            input_key:              The key your graph expects for input.
            output_key:             The key your graph returns output on.
            graph_config:           Optional config dict passed to graph.invoke().
        """
        self.graph = graph
        self.paraphraser = LightweightParaphraser(llm_callable, n_variants)
        self.scorer = SemanticScorer()
        self.threshold = consistency_threshold
        self.input_key = input_key
        self.output_key = output_key
        self.graph_config = graph_config or {}

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(self, query: str) -> ConsistencyReport:
        """
        Run consistency evaluation on a single query.

        Steps:
          1. Get original agent response
          2. Generate paraphrased variants
          3. Get agent response for each variant
          4. Score semantic similarity of all responses to the original
          5. Compute consistency score and flag if below threshold
        """
        eval_start = time.time()

        # Step 1: original response
        original_response, original_time = self._invoke_graph(query)

        # Step 2: generate variants
        variants = self.paraphraser.generate(query)

        # Step 3: get responses for all variants
        variant_results: list[VariantResult] = []
        variant_responses: list[str] = []

        for variant_query in variants:
            response, response_time = self._invoke_graph(variant_query)
            variant_results.append(VariantResult(
                variant_query=variant_query,
                agent_response=response,
                response_time_ms=response_time,
            ))
            variant_responses.append(response)

        # Step 4: score all variant responses against original
        similarities = self.scorer.score_many(original_response, variant_responses)
        for result, sim in zip(variant_results, similarities):
            result.similarity_to_original = sim

        # Step 5: compute aggregate consistency score
        consistency_score = round(sum(similarities) / len(similarities), 4) if similarities else 0.0
        min_sim = round(min(similarities), 4) if similarities else 0.0
        max_sim = round(max(similarities), 4) if similarities else 0.0
        flagged = consistency_score < self.threshold

        eval_duration_ms = round((time.time() - eval_start) * 1000, 2)

        return ConsistencyReport(
            original_query=query,
            original_response=original_response,
            consistency_score=consistency_score,
            min_similarity=min_sim,
            max_similarity=max_sim,
            flagged=flagged,
            flag_reason=(
                f"Consistency score {consistency_score:.2f} is below threshold {self.threshold:.2f}. "
                f"The agent produced semantically inconsistent responses to equivalent queries. "
                f"Lowest similarity: {min_sim:.2f}."
            ) if flagged else None,
            variants=variant_results,
            num_variants=len(variant_results),
            evaluation_duration_ms=eval_duration_ms,
        )

    def evaluate_batch(self, queries: list[str]) -> list[ConsistencyReport]:
        """Evaluate a list of queries. Returns one report per query."""
        return [self.evaluate(q) for q in queries]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _invoke_graph(self, query: str) -> tuple[str, float]:
        """Invoke the LangGraph graph and return (response_text, duration_ms)."""
        start = time.time()
        state = {self.input_key: [HumanMessage(content=query)]}
        result = self.graph.invoke(state, config=self.graph_config)
        duration_ms = round((time.time() - start) * 1000, 2)
        response_text = self._extract_response(result)
        return response_text, duration_ms

    def _extract_response(self, result: Any) -> str:
        """Extract text response from LangGraph output."""
        # Handle standard messages output
        if isinstance(result, dict):
            messages = result.get(self.output_key, [])
            if messages:
                last = messages[-1]
                if hasattr(last, "content"):
                    return str(last.content)
                if isinstance(last, dict):
                    return str(last.get("content", last))
        # Fallback: stringify the result
        return str(result)
