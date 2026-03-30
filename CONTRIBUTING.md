# Contributing to AgentMirror

First — thank you. AgentMirror exists because production multi-agent systems
have real evaluation problems that nobody has solved cleanly. If you've hit
those problems, you're exactly who this project is for.

---

## The Spirit of This Project

AgentMirror is built from real pain, not academic interest. Every module
in this library traces back to a specific failure mode encountered while
running agents in production. Contributions should follow that same
principle: **if it solves a real problem you've faced, it belongs here.**

---

## Ways to Contribute

You don't have to write code to contribute meaningfully.

**Report a real failure mode** — if you've run multi-agent systems in
production and hit an evaluation problem that AgentMirror doesn't cover,
open an issue and describe it. The more specific the better. "I had no way
to know which agent caused the failure" is more useful than "evaluation
is hard."

**Improve the docs** — if something in the README or docstrings was
confusing, fix it. Clear documentation is not a minor contribution.

**Add a real-world example** — if you've used ConsistencyJudge (or any
future module) on a real agent and have results to share, add an example
to the `examples/` folder. Sanitize any sensitive data, keep the
structure honest.

**Fix a bug** — if something broke for you, it will break for others.

**Build a new module** — if you have a concrete proposal for ChainBlame,
ReasoningScorer, or JudgeCalibrator, open an issue first to discuss the
design before writing code.

---

## Before You Open a PR

**1. Open an issue first for anything non-trivial.**
If you're fixing a typo or a one-line bug, just send the PR. For anything
larger — a new feature, a new module, a significant refactor — open an
issue first and describe what you're building and why. This saves both
of us time.

**2. Make sure tests pass.**
```bash
pip install -e ".[dev]"
pytest tests/
```
All existing tests must pass. New features need new tests.

**3. Keep the scope tight.**
One PR, one thing. A PR that fixes a bug AND adds a feature AND refactors
a module is hard to review and hard to reason about. Split them.

---

## How to Set Up Locally

```bash
# Clone the repo
git clone https://github.com/naureen-fathima/agentmirror.git
cd agentmirror

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run a specific test
pytest tests/test_consistency_judge.py::TestConsistencyJudge::test_consistent_agent_scores_high
```

---

## Project Structure

```
agentmirror/
├── agentmirror/
│   ├── __init__.py              # public API — keep this clean
│   └── consistency_judge.py     # ConsistencyJudge module
├── examples/
│   └── basic_usage.py           # runnable examples
├── tests/
│   └── test_consistency_judge.py
├── CONTRIBUTING.md              # you are here
├── LICENSE
├── README.md
└── pyproject.toml
```

When adding a new module (e.g. `chain_blame.py`), follow the same pattern:
- one file per module in `agentmirror/`
- export it from `__init__.py`
- tests in `tests/test_<module_name>.py`
- at least one example in `examples/`

---

## Code Style

- Python 3.10+
- Type hints everywhere — this is a library, not a script
- Pydantic models for all structured outputs
- Docstrings on every public class and method
- No external dependencies beyond what's in `pyproject.toml` unless
  genuinely necessary and discussed in an issue first

We don't have a linter enforced yet. Use common sense. Write code you'd
want to read at 11pm when something is broken in production.

---

## Commit Messages

Keep them honest and specific.

```
# Good
feat: add fallback paraphraser when LLM call fails
fix: semantic scorer returns wrong similarity for empty strings
docs: clarify consistency_threshold behaviour in README

# Not useful
update code
fix stuff
wip
```

---

## Roadmap Modules (Open for Discussion)

These are the next three modules on the roadmap. If you want to build
one, open an issue — don't start coding in isolation.

| Module | Problem it solves |
|---|---|
| `ChainBlame` | Traces a failure to the specific agent in a multi-hop chain |
| `ReasoningScorer` | Scores quality of reasoning, not just output text |
| `JudgeCalibrator` | Measures how much you can trust your LLM-as-judge |

---

## A Note on Scope

AgentMirror is deliberately narrow. It is not a general LLM observability
platform. It is not a prompt management tool. It is not a dashboard.

It is an evaluation library for multi-agent systems that operate without
ground truth. Contributions that expand the scope significantly will be
discussed carefully before merging — not because the ideas are bad, but
because focus is how small open source projects stay useful.

---

## Questions

Open an issue. Tag it `question`. There are no stupid questions if you've
read this document first.

---

*AgentMirror is maintained by [Naureen Fathima](https://github.com/naureen-fathima).*
*Built from real production pain. Contributions welcome from anyone who's felt it.*