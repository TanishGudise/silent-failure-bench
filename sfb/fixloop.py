"""Fix loop: propose a config fix from a triggering trace, then check it.

Seeds (pre-registered): bench 0-999; gate held-out 5000-5004, outcome
held-out 5005-5024 (same family, induced); gate regression 6000-6004, outcome
regression 6005-6024 (other families, alternating induced/natural). Replay is
the triggering task itself, run 3 times with the fixed config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .agent import TOOL_SCHEMAS, AgentConfig
from .detectors import judge_input
from .labels import SILENT
from .tasks import FAMILIES, Task, build

ISSUE_TITLES = {
    "wrong_entity": "Agent cancels or changes a reservation the customer didn't ask about",
    "stale_context": "Agent acts on a date the customer had already corrected",
    "lost_context": "Agent ignores how the customer asked to be refunded",
    "loop": "Agent repeats the same failing tool call over and over",
    "false_success": "Agent tells the customer a cancellation is done when it isn't",
    "partial": "Agent reports a multi-part request as done when one part failed",
}

FIXER_SYSTEM = """You maintain a production customer-support agent. You get an issue with a trace that shows it, plus the agent's current instructions and tool descriptions. Propose the smallest change to the instructions and/or tool descriptions that fixes the issue without breaking other behavior. You can't change code or tools, only their text."""

FIXER_USER = """Issue: {title}

Current instructions:
<instructions>
{instructions}
</instructions>

Current tool descriptions:
<tools>
{tools}
</tools>

Trace showing the issue (JSON):
<trace>
{trace}
</trace>

Return the complete new instructions, and new text for any tool description you change."""

FIXER_SCHEMA = {
    "type": "object",
    "properties": {
        "instructions": {"type": "string"},
        "tool_changes": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string", "enum": list(TOOL_SCHEMAS)},
                           "description": {"type": "string"}},
            "required": ["name", "description"], "additionalProperties": False}},
        "rationale": {"type": "string"},
    },
    "required": ["instructions", "tool_changes", "rationale"],
    "additionalProperties": False,
}

REPLAYS = 3
GATE_HELDOUT = range(5000, 5005)
OUTCOME_HELDOUT = range(5005, 5025)
GATE_REGRESSION = range(6000, 6005)
OUTCOME_REGRESSION = range(6005, 6025)
# Outcome thresholds on 20 paired seeds each (pre-registered).
FIXED_DROP = 3
REGRESS_RISE = 3


def fixer_prompt(config: AgentConfig, family: str, trigger_trace: dict[str, Any]) -> tuple[str, str]:
    tools = "\n".join(f"- {name}: {text}" for name, text in config.tool_descriptions)
    return FIXER_SYSTEM, FIXER_USER.format(title=ISSUE_TITLES[family], instructions=config.system_prompt,
                                           tools=tools, trace=judge_input(trigger_trace))


def apply_fix(config: AgentConfig, answer: dict[str, Any]) -> AgentConfig:
    changes = {c["name"]: c["description"] for c in answer.get("tool_changes") or []
               if isinstance(c, dict) and c.get("name") in TOOL_SCHEMAS}
    return config.with_edits(system_prompt=answer.get("instructions") or None, tool_descriptions=changes)


def heldout_tasks(family: str, seeds: range) -> list[Task]:
    return [build(family, s, True) for s in seeds]


def regression_tasks(family: str, seeds: range) -> list[Task]:
    others = [f for f in FAMILIES if f != family]
    return [build(others[i % len(others)], s, i % 2 == 0) for i, s in enumerate(seeds)]


def failed(label: str) -> bool:
    return label in SILENT


@dataclass
class FixResult:
    family: str
    trigger: str
    replay_labels: list[str]
    gate_heldout: tuple[int, int]      # (baseline fails, fixed fails)
    gate_regression: tuple[int, int]
    outcome_heldout: tuple[int, int]
    outcome_regression: tuple[int, int]

    @property
    def replay_pass(self) -> bool:
        return all(not failed(label) for label in self.replay_labels)

    @property
    def outcome(self) -> str:
        bh, fh = self.outcome_heldout
        br, fr = self.outcome_regression
        if fr >= br + REGRESS_RISE or fh >= bh + REGRESS_RISE:
            return "regressed"
        if fh <= bh - FIXED_DROP and fr < br + REGRESS_RISE:
            return "fixed"
        return "unchanged"

    def gate(self, name: str) -> bool:
        g1 = self.replay_pass
        g2 = g1 and self.gate_heldout[1] < self.gate_heldout[0]
        g3 = g2 and self.gate_regression[1] <= self.gate_regression[0]
        return {"G1": g1, "G2": g2, "G3": g3}[name]

    def record(self) -> dict[str, Any]:
        return {**self.__dict__, "replay_pass": self.replay_pass, "outcome": self.outcome,
                "gates": {g: self.gate(g) for g in ("G1", "G2", "G3")}}


def gate_table(results: list[FixResult]) -> dict[str, dict[str, int]]:
    """Per gate: PRs it would open, how many of them are bad, good fixes it keeps or blocks."""
    table = {}
    for gate in ("G1", "G2", "G3"):
        opened = [r for r in results if r.gate(gate)]
        table[gate] = {
            "opened": len(opened),
            "bad_opened": sum(r.outcome != "fixed" for r in opened),
            "good_kept": sum(r.outcome == "fixed" for r in opened),
            "good_blocked": sum(r.outcome == "fixed" and not r.gate(gate) for r in results),
        }
    return table


def replay_false_pass(results: list[FixResult]) -> tuple[int, int]:
    """(replay-passing fixes that are unchanged or regressed, replay-passing fixes)."""
    passing = [r for r in results if r.replay_pass]
    return sum(r.outcome != "fixed" for r in passing), len(passing)


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o))
